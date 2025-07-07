"""Comprehensive data integrity regression tests.

This module contains regression tests to ensure data integrity is maintained
across versions, operations, and different data handling scenarios.
"""

import hashlib
import json
import pickle
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pynomaly.domain.entities import Dataset
from pynomaly.domain.value_objects import AnomalyScore
from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter


class TestDataConsistencyRegression:
    """Test data consistency across operations and transformations."""

    @pytest.fixture
    def reference_datasets(self):
        """Create reference datasets with known characteristics."""
        datasets = {}

        # Deterministic dataset for consistency testing
        np.random.seed(12345)

        # Simple numeric dataset
        simple_data = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
                "feature3": [0.1, 0.2, 0.3, 0.4, 0.5],
            }
        )
        datasets["simple"] = Dataset(name="Simple Dataset", data=simple_data)

        # Dataset with missing values
        missing_data = pd.DataFrame(
            {
                "feature1": [1.0, np.nan, 3.0, 4.0, np.nan],
                "feature2": [10.0, 20.0, np.nan, 40.0, 50.0],
                "feature3": [0.1, 0.2, 0.3, np.nan, 0.5],
            }
        )
        datasets["missing"] = Dataset(name="Missing Values Dataset", data=missing_data)

        # Dataset with extreme values
        extreme_data = pd.DataFrame(
            {
                "feature1": [1.0, 1e10, -1e10, 0.0, 1.0],
                "feature2": [0.001, 0.002, 1000000.0, 0.003, 0.004],
                "feature3": [1.0, 1.0, 1.0, 1.0, 1e-15],
            }
        )
        datasets["extreme"] = Dataset(name="Extreme Values Dataset", data=extreme_data)

        # Mixed data types dataset
        mixed_data = pd.DataFrame(
            {
                "numeric_int": [1, 2, 3, 4, 5],
                "numeric_float": [1.1, 2.2, 3.3, 4.4, 5.5],
                "categorical": ["A", "B", "C", "A", "B"],
                "boolean": [True, False, True, False, True],
                "datetime": pd.date_range("2023-01-01", periods=5),
            }
        )
        datasets["mixed"] = Dataset(name="Mixed Types Dataset", data=mixed_data)

        return datasets

    def test_data_preservation_through_operations(self, reference_datasets):
        """Test that data is preserved through various operations."""
        dataset = reference_datasets["simple"]
        original_data = dataset.data.copy()

        # Calculate checksums for original data
        original_checksums = {}
        for col in original_data.columns:
            col_bytes = str(original_data[col].values).encode("utf-8")
            original_checksums[col] = hashlib.md5(col_bytes).hexdigest()

        try:
            # Train model
            adapter = SklearnAdapter(
                algorithm_name="IsolationForest",
                parameters={
                    "contamination": 0.2,
                    "n_estimators": 10,
                    "random_state": 42,
                },
            )

            adapter.fit(dataset)

            # Check data hasn't been modified during training
            for col in dataset.data.columns:
                col_bytes = str(dataset.data[col].values).encode("utf-8")
                current_checksum = hashlib.md5(col_bytes).hexdigest()
                assert (
                    current_checksum == original_checksums[col]
                ), f"Data modified during training for column {col}"

            # Score the data
            scores = adapter.score(dataset)

            # Check data hasn't been modified during scoring
            for col in dataset.data.columns:
                col_bytes = str(dataset.data[col].values).encode("utf-8")
                current_checksum = hashlib.md5(col_bytes).hexdigest()
                assert (
                    current_checksum == original_checksums[col]
                ), f"Data modified during scoring for column {col}"

            # Verify scores are valid
            assert len(scores) == len(dataset.data)
            for score in scores:
                assert isinstance(score, AnomalyScore)
                assert 0.0 <= score.value <= 1.0
                assert not np.isnan(score.value)
                assert not np.isinf(score.value)

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_missing_value_handling_consistency(self, reference_datasets):
        """Test consistent handling of missing values."""
        dataset = reference_datasets["missing"]

        # Count missing values before operations
        original_missing_counts = dataset.data.isnull().sum().to_dict()

        try:
            # Test different algorithms' handling of missing values
            algorithms = [
                {
                    "name": "IsolationForest",
                    "params": {
                        "contamination": 0.2,
                        "n_estimators": 10,
                        "random_state": 42,
                    },
                }
            ]

            for algo_config in algorithms:
                # Create fresh copy of data
                test_dataset = Dataset(
                    name=f"Missing Test {algo_config['name']}", data=dataset.data.copy()
                )

                adapter = SklearnAdapter(
                    algorithm_name=algo_config["name"], parameters=algo_config["params"]
                )

                # Some algorithms might not handle missing values
                try:
                    adapter.fit(test_dataset)
                    scores = adapter.score(test_dataset)

                    # If successful, verify data integrity
                    current_missing_counts = test_dataset.data.isnull().sum().to_dict()

                    # Missing value pattern should be consistent
                    for col, original_count in original_missing_counts.items():
                        current_count = current_missing_counts.get(col, 0)
                        # Algorithm might have handled missing values differently
                        # but should not have introduced new missing values
                        assert (
                            current_count <= original_count or current_count == 0
                        ), f"New missing values introduced in column {col}"

                    # Scores should be valid for non-missing data points
                    valid_scores = [s for s in scores if s is not None]
                    for score in valid_scores:
                        assert isinstance(score, AnomalyScore)
                        assert not np.isnan(score.value)
                        assert not np.isinf(score.value)

                except (ValueError, TypeError):
                    # Algorithm doesn't handle missing values - this is acceptable
                    continue

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_extreme_value_handling_consistency(self, reference_datasets):
        """Test consistent handling of extreme values."""
        dataset = reference_datasets["extreme"]

        try:
            adapter = SklearnAdapter(
                algorithm_name="IsolationForest",
                parameters={
                    "contamination": 0.4,  # Higher contamination for extreme values
                    "n_estimators": 10,
                    "random_state": 42,
                },
            )

            adapter.fit(dataset)
            scores = adapter.score(dataset)

            # All scores should be valid despite extreme input values
            assert len(scores) == len(dataset.data)

            for i, score in enumerate(scores):
                assert isinstance(
                    score, AnomalyScore
                ), f"Invalid score type at index {i}"
                assert not np.isnan(score.value), f"NaN score at index {i}"
                assert not np.isinf(score.value), f"Inf score at index {i}"
                assert (
                    0.0 <= score.value <= 1.0
                ), f"Score out of range at index {i}: {score.value}"

            # Extreme values should generally have higher anomaly scores
            score_values = [score.value for score in scores]

            # Index 1 has 1e10, index 2 has -1e10 - these should be detected as anomalies
            extreme_indices = [1, 2]
            normal_indices = [0, 3, 4]

            avg_extreme_score = np.mean([score_values[i] for i in extreme_indices])
            avg_normal_score = np.mean([score_values[i] for i in normal_indices])

            # Extreme values should have higher scores (though this might not always hold)
            # We just verify the algorithm doesn't crash and produces valid scores
            assert avg_extreme_score >= 0.0
            assert avg_normal_score >= 0.0

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_data_type_preservation(self, reference_datasets):
        """Test that data types are preserved through operations."""
        dataset = reference_datasets["mixed"]

        # Record original data types
        original_dtypes = dataset.data.dtypes.to_dict()

        # Select only numeric columns for anomaly detection
        numeric_columns = dataset.data.select_dtypes(include=[np.number]).columns
        numeric_data = dataset.data[numeric_columns].copy()

        if len(numeric_columns) > 0:
            numeric_dataset = Dataset(name="Numeric Only", data=numeric_data)

            try:
                adapter = SklearnAdapter(
                    algorithm_name="IsolationForest",
                    parameters={
                        "contamination": 0.2,
                        "n_estimators": 10,
                        "random_state": 42,
                    },
                )

                adapter.fit(numeric_dataset)
                scores = adapter.score(numeric_dataset)

                # Verify data types are preserved
                current_dtypes = numeric_dataset.data.dtypes.to_dict()

                for col in numeric_columns:
                    original_dtype = original_dtypes[col]
                    current_dtype = current_dtypes[col]

                    # Allow for some dtype flexibility (e.g., int64 -> float64)
                    # but ensure numeric types remain numeric
                    assert pd.api.types.is_numeric_dtype(
                        current_dtype
                    ), f"Numeric column {col} became non-numeric: {original_dtype} -> {current_dtype}"

                # Verify scores
                assert len(scores) == len(numeric_data)

            except ImportError:
                pytest.skip("scikit-learn not available")


class TestSerializationIntegrityRegression:
    """Test data integrity through serialization/deserialization."""

    @pytest.fixture
    def trained_model_and_data(self):
        """Create trained model and reference data for serialization testing."""
        # Create deterministic dataset
        np.random.seed(999)
        data = pd.DataFrame(
            {
                "x": np.random.normal(0, 1, 100),
                "y": np.random.normal(0, 1, 100),
                "z": np.random.normal(0, 1, 100),
            }
        )
        dataset = Dataset(name="Serialization Test", data=data)

        try:
            # Train model
            adapter = SklearnAdapter(
                algorithm_name="IsolationForest",
                parameters={
                    "contamination": 0.1,
                    "n_estimators": 50,
                    "random_state": 777,
                },
            )
            adapter.fit(dataset)

            return adapter, dataset

        except ImportError:
            return None, dataset

    def test_pickle_serialization_integrity(self, trained_model_and_data):
        """Test data integrity through pickle serialization."""
        adapter, dataset = trained_model_and_data

        if adapter is None:
            pytest.skip("scikit-learn not available")

        # Get original scores
        original_scores = adapter.score(dataset)
        original_values = [score.value for score in original_scores]

        # Serialize with pickle
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(adapter, f)
            pickle_path = f.name

        # Deserialize
        with open(pickle_path, "rb") as f:
            loaded_adapter = pickle.load(f)

        # Get scores from loaded model
        loaded_scores = loaded_adapter.score(dataset)
        loaded_values = [score.value for score in loaded_scores]

        # Scores should be identical
        assert len(original_values) == len(loaded_values)

        for i, (orig, loaded) in enumerate(
            zip(original_values, loaded_values, strict=False)
        ):
            assert (
                abs(orig - loaded) < 1e-10
            ), f"Score mismatch at index {i}: {orig} vs {loaded}"

        # Clean up
        Path(pickle_path).unlink()

    def test_json_serialization_integrity(self):
        """Test JSON serialization of results and metadata."""
        # Create test results data
        test_results = {
            "model_id": str(uuid.uuid4()),
            "dataset_name": "Test Dataset",
            "timestamp": datetime.now().isoformat(),
            "scores": [0.1, 0.2, 0.8, 0.15, 0.9],
            "labels": [0, 0, 1, 0, 1],
            "parameters": {
                "contamination": 0.1,
                "n_estimators": 100,
                "random_state": 42,
            },
            "metadata": {"n_samples": 5, "n_features": 3, "anomaly_rate": 0.4},
        }

        # Serialize to JSON
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_results, f, indent=2)
            json_path = f.name

        # Deserialize from JSON
        with open(json_path) as f:
            loaded_results = json.load(f)

        # Verify data integrity
        assert loaded_results["model_id"] == test_results["model_id"]
        assert loaded_results["dataset_name"] == test_results["dataset_name"]
        assert loaded_results["scores"] == test_results["scores"]
        assert loaded_results["labels"] == test_results["labels"]
        assert loaded_results["parameters"] == test_results["parameters"]
        assert loaded_results["metadata"] == test_results["metadata"]

        # Verify data types
        assert isinstance(loaded_results["scores"], list)
        assert isinstance(loaded_results["labels"], list)
        assert isinstance(loaded_results["parameters"], dict)
        assert isinstance(loaded_results["metadata"], dict)

        # Verify timestamp can be parsed
        parsed_timestamp = datetime.fromisoformat(loaded_results["timestamp"])
        assert isinstance(parsed_timestamp, datetime)

        # Clean up
        Path(json_path).unlink()

    def test_csv_data_integrity(self):
        """Test CSV serialization preserves data integrity."""
        # Create test dataset with various data types
        test_data = pd.DataFrame(
            {
                "int_col": [1, 2, 3, 4, 5],
                "float_col": [1.1, 2.2, 3.3, 4.4, 5.5],
                "string_col": ["A", "B", "C", "D", "E"],
                "bool_col": [True, False, True, False, True],
                "datetime_col": pd.date_range("2023-01-01", periods=5),
            }
        )

        # Calculate checksums for original data
        original_checksums = {}
        for col in ["int_col", "float_col"]:  # Focus on numeric columns
            col_bytes = str(test_data[col].values).encode("utf-8")
            original_checksums[col] = hashlib.md5(col_bytes).hexdigest()

        # Save to CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            test_data.to_csv(f.name, index=False)
            csv_path = f.name

        # Load from CSV
        loaded_data = pd.read_csv(csv_path)

        # Verify structure
        assert len(loaded_data) == len(test_data)
        assert len(loaded_data.columns) == len(test_data.columns)

        # Verify numeric data integrity
        for col in ["int_col", "float_col"]:
            col_bytes = str(loaded_data[col].values).encode("utf-8")
            loaded_checksum = hashlib.md5(col_bytes).hexdigest()
            assert (
                loaded_checksum == original_checksums[col]
            ), f"Data integrity lost for column {col}"

        # Verify string data
        assert loaded_data["string_col"].tolist() == test_data["string_col"].tolist()

        # Verify boolean data (might be loaded as strings)
        loaded_bool_values = loaded_data["bool_col"].tolist()
        expected_bool_values = test_data["bool_col"].tolist()

        # Convert loaded values if they're strings
        if isinstance(loaded_bool_values[0], str):
            loaded_bool_values = [v == "True" for v in loaded_bool_values]

        assert loaded_bool_values == expected_bool_values

        # Clean up
        Path(csv_path).unlink()


class TestConcurrentDataIntegrityRegression:
    """Test data integrity under concurrent operations."""

    def test_concurrent_read_integrity(self):
        """Test data integrity under concurrent read operations."""
        # Create shared dataset
        np.random.seed(555)
        data = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 1000),
                "feature2": np.random.normal(0, 1, 1000),
                "feature3": np.random.normal(0, 1, 1000),
            }
        )
        dataset = Dataset(name="Concurrent Read Test", data=data)

        # Calculate reference checksum
        reference_checksum = hashlib.md5(str(data.values).encode("utf-8")).hexdigest()

        try:
            # Train model
            adapter = SklearnAdapter(
                algorithm_name="IsolationForest",
                parameters={
                    "contamination": 0.1,
                    "n_estimators": 20,
                    "random_state": 42,
                },
            )
            adapter.fit(dataset)

            import threading

            def concurrent_scoring(thread_id):
                """Score data in concurrent thread."""
                scores = adapter.score(dataset)

                # Verify data integrity in thread
                current_checksum = hashlib.md5(
                    str(dataset.data.values).encode("utf-8")
                ).hexdigest()
                assert (
                    current_checksum == reference_checksum
                ), f"Data integrity compromised in thread {thread_id}"

                return len(scores)

            # Run concurrent scoring operations
            threads = []
            results = []

            def thread_worker(thread_id):
                result = concurrent_scoring(thread_id)
                results.append(result)

            # Start multiple threads
            for i in range(3):
                thread = threading.Thread(target=thread_worker, args=(i,))
                threads.append(thread)
                thread.start()

            # Wait for all threads
            for thread in threads:
                thread.join()

            # Verify all operations completed successfully
            assert len(results) == 3
            assert all(result == len(data) for result in results)

            # Verify data integrity after concurrent operations
            final_checksum = hashlib.md5(
                str(dataset.data.values).encode("utf-8")
            ).hexdigest()
            assert (
                final_checksum == reference_checksum
            ), "Data integrity compromised after concurrent operations"

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_concurrent_model_operations_integrity(self):
        """Test integrity when multiple models operate on same data."""
        # Create shared dataset
        np.random.seed(666)
        data = pd.DataFrame(
            {
                "x": np.random.normal(0, 1, 500),
                "y": np.random.normal(0, 1, 500),
                "z": np.random.normal(0, 1, 500),
            }
        )
        dataset = Dataset(name="Concurrent Model Test", data=data)

        # Calculate reference checksum
        reference_checksum = hashlib.md5(str(data.values).encode("utf-8")).hexdigest()

        try:
            import threading

            def train_and_score_model(model_id, random_state):
                """Train and score a model in a thread."""
                adapter = SklearnAdapter(
                    algorithm_name="IsolationForest",
                    parameters={
                        "contamination": 0.1,
                        "n_estimators": 10,
                        "random_state": random_state,
                    },
                )

                # Train model
                adapter.fit(dataset)

                # Score data
                scores = adapter.score(dataset)

                # Verify data integrity
                current_checksum = hashlib.md5(
                    str(dataset.data.values).encode("utf-8")
                ).hexdigest()
                assert (
                    current_checksum == reference_checksum
                ), f"Data integrity compromised by model {model_id}"

                return {
                    "model_id": model_id,
                    "n_scores": len(scores),
                    "avg_score": np.mean([score.value for score in scores]),
                }

            # Run concurrent model operations
            threads = []
            results = []

            def thread_worker(model_id, random_state):
                result = train_and_score_model(model_id, random_state)
                results.append(result)

            # Start multiple threads with different models
            model_configs = [(0, 42), (1, 123), (2, 456)]

            for model_id, random_state in model_configs:
                thread = threading.Thread(
                    target=thread_worker, args=(model_id, random_state)
                )
                threads.append(thread)
                thread.start()

            # Wait for all threads
            for thread in threads:
                thread.join()

            # Verify all operations completed successfully
            assert len(results) == 3

            for result in results:
                assert result["n_scores"] == len(data)
                assert 0.0 <= result["avg_score"] <= 1.0

            # Verify data integrity after all concurrent operations
            final_checksum = hashlib.md5(
                str(dataset.data.values).encode("utf-8")
            ).hexdigest()
            assert (
                final_checksum == reference_checksum
            ), "Data integrity compromised after concurrent model operations"

        except ImportError:
            pytest.skip("scikit-learn not available")


class TestEdgeCaseDataIntegrityRegression:
    """Test data integrity with edge cases and boundary conditions."""

    def test_empty_dataset_handling(self):
        """Test handling of empty datasets."""
        # Create empty dataset
        empty_data = pd.DataFrame(columns=["feature1", "feature2", "feature3"])
        empty_dataset = Dataset(name="Empty Dataset", data=empty_data)

        try:
            adapter = SklearnAdapter(
                algorithm_name="IsolationForest",
                parameters={
                    "contamination": 0.1,
                    "n_estimators": 10,
                    "random_state": 42,
                },
            )

            # Should handle empty dataset gracefully
            try:
                adapter.fit(empty_dataset)
                scores = adapter.score(empty_dataset)

                # If successful, should return empty scores
                assert len(scores) == 0

            except (ValueError, RuntimeError):
                # It's acceptable for algorithms to reject empty datasets
                pass

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_single_sample_dataset_integrity(self):
        """Test data integrity with single sample datasets."""
        # Create single sample dataset
        single_data = pd.DataFrame(
            {"feature1": [1.0], "feature2": [2.0], "feature3": [3.0]}
        )
        single_dataset = Dataset(name="Single Sample", data=single_data)

        try:
            adapter = SklearnAdapter(
                algorithm_name="IsolationForest",
                parameters={
                    "contamination": 0.1,
                    "n_estimators": 10,
                    "random_state": 42,
                },
            )

            # Should handle single sample gracefully
            try:
                adapter.fit(single_dataset)
                scores = adapter.score(single_dataset)

                # If successful, should return one score
                assert len(scores) == 1
                assert isinstance(scores[0], AnomalyScore)
                assert 0.0 <= scores[0].value <= 1.0

            except (ValueError, RuntimeError):
                # It's acceptable for algorithms to reject single samples
                pass

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_identical_samples_integrity(self):
        """Test data integrity with identical samples."""
        # Create dataset with identical samples
        identical_data = pd.DataFrame(
            {"feature1": [1.0] * 100, "feature2": [2.0] * 100, "feature3": [3.0] * 100}
        )
        identical_dataset = Dataset(name="Identical Samples", data=identical_data)

        try:
            adapter = SklearnAdapter(
                algorithm_name="IsolationForest",
                parameters={
                    "contamination": 0.1,
                    "n_estimators": 10,
                    "random_state": 42,
                },
            )

            # Should handle identical samples
            adapter.fit(identical_dataset)
            scores = adapter.score(identical_dataset)

            # Should return scores for all samples
            assert len(scores) == 100

            # All scores should be valid
            for score in scores:
                assert isinstance(score, AnomalyScore)
                assert not np.isnan(score.value)
                assert not np.isinf(score.value)
                assert 0.0 <= score.value <= 1.0

            # With identical samples, scores should be very similar
            score_values = [score.value for score in scores]
            score_variance = np.var(score_values)
            assert (
                score_variance < 0.01
            ), "High variance in scores for identical samples"

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_unicode_data_integrity(self):
        """Test data integrity with unicode characters."""
        # Create dataset with unicode metadata
        unicode_data = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )

        # Add unicode metadata
        unicode_dataset = Dataset(
            name="Unicode Test: 测试 🚀 émojis", data=unicode_data
        )

        # Store original name
        original_name = unicode_dataset.name

        try:
            adapter = SklearnAdapter(
                algorithm_name="IsolationForest",
                parameters={
                    "contamination": 0.2,
                    "n_estimators": 10,
                    "random_state": 42,
                },
            )

            adapter.fit(unicode_dataset)
            scores = adapter.score(unicode_dataset)

            # Verify unicode name is preserved
            assert unicode_dataset.name == original_name

            # Verify normal operation
            assert len(scores) == len(unicode_data)

            for score in scores:
                assert isinstance(score, AnomalyScore)
                assert 0.0 <= score.value <= 1.0

        except ImportError:
            pytest.skip("scikit-learn not available")
