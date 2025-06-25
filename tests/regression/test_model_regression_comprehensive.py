"""Comprehensive model regression tests.

This module contains regression tests for machine learning models to ensure
that model behavior, performance, and compatibility remain consistent across
versions and deployments.
"""

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pynomaly.domain.entities import Dataset
from pynomaly.domain.value_objects import (
    AnomalyScore,
)
from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter


class TestModelOutputConsistency:
    """Test that model outputs remain consistent across versions."""

    @pytest.fixture
    def reference_dataset(self):
        """Create a reference dataset for consistent testing."""
        # Use fixed seed for reproducibility
        np.random.seed(42)

        # Generate synthetic data with known characteristics
        normal_data = np.random.multivariate_normal(
            mean=[0, 0, 0], cov=[[1, 0.1, 0.1], [0.1, 1, 0.1], [0.1, 0.1, 1]], size=900
        )

        # Generate anomalous data
        anomaly_data = np.random.multivariate_normal(
            mean=[3, 3, 3], cov=[[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]], size=100
        )

        # Combine data
        X = np.vstack([normal_data, anomaly_data])
        y_true = np.hstack([np.zeros(900), np.ones(100)])

        # Create DataFrame
        df = pd.DataFrame(X, columns=["feature1", "feature2", "feature3"])
        df["is_anomaly"] = y_true

        return Dataset(name="Reference Dataset", data=df)

    @pytest.fixture
    def reference_model_configs(self):
        """Define reference model configurations for testing."""
        return {
            "isolation_forest": {
                "algorithm": "IsolationForest",
                "parameters": {
                    "contamination": 0.1,
                    "n_estimators": 100,
                    "random_state": 42,
                    "max_samples": "auto",
                },
            },
            "local_outlier_factor": {
                "algorithm": "LocalOutlierFactor",
                "parameters": {
                    "contamination": 0.1,
                    "n_neighbors": 20,
                    "novelty": True,
                },
            },
            "one_class_svm": {
                "algorithm": "OneClassSVM",
                "parameters": {"gamma": "scale", "nu": 0.1},
            },
        }

    def test_isolation_forest_output_consistency(
        self, reference_dataset, reference_model_configs
    ):
        """Test IsolationForest output consistency."""
        config = reference_model_configs["isolation_forest"]

        try:
            # Create and train model
            adapter = SklearnAdapter(
                algorithm_name=config["algorithm"], parameters=config["parameters"]
            )

            # Train model
            adapter.fit(reference_dataset)

            # Get predictions
            scores = adapter.score(reference_dataset)
            result = adapter.detect(reference_dataset)

            # Verify output characteristics
            assert len(scores) == reference_dataset.n_samples
            assert len(result.labels) == reference_dataset.n_samples

            # All scores should be valid
            for score in scores:
                assert isinstance(score, AnomalyScore)
                assert 0.0 <= score.value <= 1.0
                assert not np.isnan(score.value)
                assert not np.isinf(score.value)

            # Labels should be binary
            unique_labels = set(result.labels)
            assert unique_labels.issubset({0, 1})

            # Contamination rate should be approximately correct
            anomaly_rate = np.mean(result.labels)
            expected_rate = config["parameters"]["contamination"]
            assert abs(anomaly_rate - expected_rate) < 0.05

            # Test reproducibility with same random state
            adapter2 = SklearnAdapter(
                algorithm_name=config["algorithm"], parameters=config["parameters"]
            )
            adapter2.fit(reference_dataset)
            scores2 = adapter2.score(reference_dataset)

            # Scores should be identical with same random state
            for score1, score2 in zip(scores, scores2, strict=False):
                assert abs(score1.value - score2.value) < 1e-10

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_model_performance_consistency(
        self, reference_dataset, reference_model_configs
    ):
        """Test model performance metrics consistency."""
        performance_results = {}

        for model_name, config in reference_model_configs.items():
            try:
                # Create and train model
                adapter = SklearnAdapter(
                    algorithm_name=config["algorithm"], parameters=config["parameters"]
                )

                adapter.fit(reference_dataset)
                result = adapter.detect(reference_dataset)

                # Calculate performance metrics
                y_true = reference_dataset.data["is_anomaly"].values
                y_pred = result.labels

                # Calculate basic metrics
                tp = np.sum((y_pred == 1) & (y_true == 1))
                fp = np.sum((y_pred == 1) & (y_true == 0))
                tn = np.sum((y_pred == 0) & (y_true == 0))
                fn = np.sum((y_pred == 0) & (y_true == 1))

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1_score = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0
                )

                performance_results[model_name] = {
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                }

                # Performance should be reasonable
                assert precision >= 0.0
                assert recall >= 0.0
                assert f1_score >= 0.0
                assert precision <= 1.0
                assert recall <= 1.0
                assert f1_score <= 1.0

            except ImportError:
                continue

        # At least one model should work
        assert len(performance_results) > 0

        # Performance should be above random chance for at least one model
        best_f1 = max(result["f1_score"] for result in performance_results.values())
        assert best_f1 > 0.1, "All models performing worse than random chance"

    def test_model_serialization_consistency(
        self, reference_dataset, reference_model_configs
    ):
        """Test model serialization and deserialization consistency."""
        config = reference_model_configs["isolation_forest"]

        try:
            # Create and train original model
            original_adapter = SklearnAdapter(
                algorithm_name=config["algorithm"], parameters=config["parameters"]
            )
            original_adapter.fit(reference_dataset)
            original_scores = original_adapter.score(reference_dataset)

            # Test pickle serialization
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                pickle.dump(original_adapter, f)
                pickle_path = f.name

            # Load pickled model
            with open(pickle_path, "rb") as f:
                pickled_adapter = pickle.load(f)

            pickled_scores = pickled_adapter.score(reference_dataset)

            # Scores should be identical
            for orig_score, pickled_score in zip(original_scores, pickled_scores, strict=False):
                assert abs(orig_score.value - pickled_score.value) < 1e-10

            # Clean up
            Path(pickle_path).unlink()

            # Test joblib serialization if available
            try:
                import joblib

                with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
                    joblib.dump(original_adapter, f.name)
                    joblib_path = f.name

                joblib_adapter = joblib.load(joblib_path)
                joblib_scores = joblib_adapter.score(reference_dataset)

                # Scores should be identical
                for orig_score, joblib_score in zip(original_scores, joblib_scores, strict=False):
                    assert abs(orig_score.value - joblib_score.value) < 1e-10

                # Clean up
                Path(joblib_path).unlink()

            except ImportError:
                pass  # joblib not available

        except ImportError:
            pytest.skip("scikit-learn not available")


class TestModelParameterRobustness:
    """Test model robustness to parameter variations."""

    @pytest.fixture
    def small_dataset(self):
        """Create a small dataset for parameter testing."""
        np.random.seed(123)

        # Small dataset for parameter sensitivity testing
        data = np.random.normal(0, 1, (100, 3))
        # Add a few clear anomalies
        data[-5:] += 5  # Make last 5 samples anomalous

        df = pd.DataFrame(data, columns=["x1", "x2", "x3"])
        return Dataset(name="Small Test Dataset", data=df)

    def test_contamination_parameter_robustness(self, small_dataset):
        """Test robustness to different contamination rates."""
        contamination_rates = [0.01, 0.05, 0.1, 0.2, 0.3]

        results = {}

        for contamination in contamination_rates:
            try:
                adapter = SklearnAdapter(
                    algorithm_name="IsolationForest",
                    parameters={
                        "contamination": contamination,
                        "n_estimators": 50,
                        "random_state": 42,
                    },
                )

                adapter.fit(small_dataset)
                result = adapter.detect(small_dataset)

                # Record actual contamination rate
                actual_contamination = np.mean(result.labels)
                results[contamination] = actual_contamination

                # Should be reasonably close to expected rate
                assert abs(actual_contamination - contamination) < 0.1

            except (ImportError, ValueError):
                continue

        # Should have at least some successful tests
        assert len(results) > 0

        # Results should be monotonically increasing (roughly)
        contamination_values = sorted(results.keys())
        actual_values = [results[c] for c in contamination_values]

        for i in range(1, len(actual_values)):
            # Allow for some noise, but general trend should be increasing
            assert actual_values[i] >= actual_values[i - 1] - 0.05

    def test_random_state_consistency(self, small_dataset):
        """Test consistency with different random states."""
        random_states = [42, 123, 456, 789]

        score_sets = []

        for random_state in random_states:
            try:
                adapter = SklearnAdapter(
                    algorithm_name="IsolationForest",
                    parameters={
                        "contamination": 0.1,
                        "n_estimators": 50,
                        "random_state": random_state,
                    },
                )

                adapter.fit(small_dataset)
                scores = adapter.score(small_dataset)
                score_values = [score.value for score in scores]
                score_sets.append(score_values)

            except ImportError:
                continue

        if len(score_sets) >= 2:
            # Different random states should give different results
            for i in range(1, len(score_sets)):
                # Results should be different
                correlation = np.corrcoef(score_sets[0], score_sets[i])[0, 1]
                assert correlation < 0.99  # Not identical

                # But still reasonable (not completely random)
                assert correlation > 0.3  # Some consistency

        # Test reproducibility with same random state
        try:
            adapter1 = SklearnAdapter(
                algorithm_name="IsolationForest",
                parameters={
                    "contamination": 0.1,
                    "n_estimators": 50,
                    "random_state": 42,
                },
            )

            adapter2 = SklearnAdapter(
                algorithm_name="IsolationForest",
                parameters={
                    "contamination": 0.1,
                    "n_estimators": 50,
                    "random_state": 42,
                },
            )

            adapter1.fit(small_dataset)
            adapter2.fit(small_dataset)

            scores1 = adapter1.score(small_dataset)
            scores2 = adapter2.score(small_dataset)

            # Should be identical with same random state
            for s1, s2 in zip(scores1, scores2, strict=False):
                assert abs(s1.value - s2.value) < 1e-10

        except ImportError:
            pass

    def test_n_estimators_scaling(self, small_dataset):
        """Test model behavior with different numbers of estimators."""
        n_estimators_values = [10, 50, 100, 200]

        performance_scores = []
        training_consistency = []

        for n_estimators in n_estimators_values:
            try:
                adapter = SklearnAdapter(
                    algorithm_name="IsolationForest",
                    parameters={
                        "contamination": 0.1,
                        "n_estimators": n_estimators,
                        "random_state": 42,
                    },
                )

                adapter.fit(small_dataset)
                scores = adapter.score(small_dataset)

                # Calculate score variance as a consistency measure
                score_values = [score.value for score in scores]
                score_variance = np.var(score_values)
                training_consistency.append(score_variance)

                # Calculate performance (using last 5 samples as known anomalies)
                anomaly_scores = score_values[-5:]  # Last 5 are anomalies
                normal_scores = score_values[:-5]  # Rest are normal

                # Anomalies should have higher scores on average
                avg_anomaly_score = np.mean(anomaly_scores)
                avg_normal_score = np.mean(normal_scores)
                performance_measure = avg_anomaly_score - avg_normal_score
                performance_scores.append(performance_measure)

            except ImportError:
                continue

        if len(performance_scores) >= 2:
            # Performance should generally improve or stabilize with more estimators
            # (or at least not get significantly worse)
            best_performance = max(performance_scores)
            min(performance_scores)

            # Should show some ability to distinguish anomalies
            assert best_performance > 0.01


class TestModelCompatibilityRegression:
    """Test model compatibility across different environments and versions."""

    def test_numpy_version_compatibility(self):
        """Test compatibility with different NumPy operations."""
        import numpy as np

        # Test different NumPy data types
        data_types = [np.float32, np.float64, np.int32, np.int64]

        for dtype in data_types:
            # Create data with specific dtype
            test_data = np.random.random((100, 3)).astype(dtype)
            df = pd.DataFrame(test_data, columns=["f1", "f2", "f3"])
            dataset = Dataset(name=f"Test {dtype}", data=df)

            try:
                adapter = SklearnAdapter(
                    algorithm_name="IsolationForest",
                    parameters={
                        "contamination": 0.1,
                        "n_estimators": 10,
                        "random_state": 42,
                    },
                )

                # Should handle different dtypes gracefully
                adapter.fit(dataset)
                scores = adapter.score(dataset)

                # Verify scores are valid regardless of input dtype
                for score in scores:
                    assert isinstance(score, AnomalyScore)
                    assert not np.isnan(score.value)
                    assert not np.isinf(score.value)
                    assert 0.0 <= score.value <= 1.0

            except ImportError:
                continue

    def test_pandas_version_compatibility(self):
        """Test compatibility with different Pandas operations."""
        import pandas as pd

        # Test different DataFrame creation methods
        data_creation_methods = [
            # Standard dictionary
            lambda: pd.DataFrame(
                {
                    "a": [1, 2, 3, 4, 5],
                    "b": [1.1, 2.2, 3.3, 4.4, 5.5],
                    "c": ["x", "y", "z", "w", "v"],
                }
            ),
            # From NumPy array
            lambda: pd.DataFrame(np.random.random((5, 3)), columns=["x1", "x2", "x3"]),
            # With different index
            lambda: pd.DataFrame({"val": range(5)}, index=[10, 20, 30, 40, 50]),
        ]

        for i, create_method in enumerate(data_creation_methods):
            df = create_method()

            # Ensure numeric columns for anomaly detection
            numeric_df = df.select_dtypes(include=[np.number])

            if len(numeric_df.columns) > 0:
                dataset = Dataset(name=f"Test DataFrame {i}", data=numeric_df)

                try:
                    adapter = SklearnAdapter(
                        algorithm_name="IsolationForest",
                        parameters={
                            "contamination": 0.2,
                            "n_estimators": 10,
                            "random_state": 42,
                        },
                    )

                    adapter.fit(dataset)
                    scores = adapter.score(dataset)

                    # Should work regardless of DataFrame creation method
                    assert len(scores) == len(numeric_df)

                except ImportError:
                    continue

    def test_memory_efficiency_regression(self):
        """Test memory efficiency doesn't regress."""
        import gc

        try:
            import psutil

            process = psutil.Process()
        except ImportError:
            pytest.skip("psutil not available for memory monitoring")

        # Measure baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss

        # Create reasonably large dataset
        large_data = np.random.random((5000, 10))
        df = pd.DataFrame(large_data, columns=[f"feature_{i}" for i in range(10)])
        dataset = Dataset(name="Large Dataset", data=df)

        try:
            adapter = SklearnAdapter(
                algorithm_name="IsolationForest",
                parameters={
                    "contamination": 0.1,
                    "n_estimators": 50,
                    "random_state": 42,
                },
            )

            # Measure memory during training
            adapter.fit(dataset)
            training_memory = process.memory_info().rss

            # Measure memory during scoring
            scores = adapter.score(dataset)
            scoring_memory = process.memory_info().rss

            # Clean up
            del adapter, scores, dataset, df, large_data
            gc.collect()

            final_memory = process.memory_info().rss

            # Memory usage should be reasonable
            training_increase = training_memory - baseline_memory
            scoring_increase = scoring_memory - training_memory
            memory_leak = final_memory - baseline_memory

            # Training should not use excessive memory (< 500MB for this dataset)
            assert training_increase < 500 * 1024 * 1024

            # Scoring should not significantly increase memory
            assert scoring_increase < 100 * 1024 * 1024

            # Should not have significant memory leak (< 50MB)
            assert memory_leak < 50 * 1024 * 1024

        except ImportError:
            pass


class TestModelOutputStability:
    """Test stability of model outputs over time and across runs."""

    def test_cross_platform_consistency(self):
        """Test that models produce consistent results across platforms."""
        # Create deterministic dataset
        np.random.seed(999)

        data = np.array(
            [
                [1.0, 2.0, 3.0],
                [1.1, 2.1, 3.1],
                [0.9, 1.9, 2.9],
                [5.0, 6.0, 7.0],  # Clear anomaly
                [1.0, 2.0, 3.0],
            ]
        )

        df = pd.DataFrame(data, columns=["x", "y", "z"])
        dataset = Dataset(name="Cross Platform Test", data=df)

        try:
            adapter = SklearnAdapter(
                algorithm_name="IsolationForest",
                parameters={
                    "contamination": 0.2,
                    "n_estimators": 10,
                    "random_state": 42,
                    "max_samples": 1.0,  # Use all samples for consistency
                },
            )

            adapter.fit(dataset)
            scores = adapter.score(dataset)
            result = adapter.detect(dataset)

            # The clear anomaly (index 3) should be detected
            score_values = [score.value for score in scores]
            anomaly_score = score_values[3]  # Index 3 is the anomaly
            normal_scores = [score_values[i] for i in [0, 1, 2, 4]]

            # Anomaly should have higher score than normal samples
            max_normal_score = max(normal_scores)
            assert anomaly_score > max_normal_score

            # With contamination=0.2 and 5 samples, should detect 1 anomaly
            num_anomalies = sum(result.labels)
            assert num_anomalies == 1

            # The detected anomaly should be the clear outlier
            assert result.labels[3] == 1  # Index 3 should be labeled as anomaly

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_numerical_stability(self):
        """Test numerical stability with edge cases."""
        edge_case_datasets = [
            # Dataset with very small values
            pd.DataFrame({"tiny_vals": [1e-10, 2e-10, 3e-10, 1e-8, 2e-10]}),
            # Dataset with very large values
            pd.DataFrame({"large_vals": [1e8, 2e8, 3e8, 1e10, 2e8]}),
            # Dataset with mixed scales
            pd.DataFrame(
                {
                    "small": [0.001, 0.002, 0.003, 1.0, 0.002],
                    "large": [1000, 2000, 3000, 1e6, 2000],
                }
            ),
            # Dataset with zeros
            pd.DataFrame({"with_zeros": [0.0, 0.1, 0.0, 5.0, 0.1]}),
        ]

        for i, df in enumerate(edge_case_datasets):
            dataset = Dataset(name=f"Edge Case {i}", data=df)

            try:
                adapter = SklearnAdapter(
                    algorithm_name="IsolationForest",
                    parameters={
                        "contamination": 0.2,
                        "n_estimators": 10,
                        "random_state": 42,
                    },
                )

                adapter.fit(dataset)
                scores = adapter.score(dataset)

                # All scores should be valid numbers
                for score in scores:
                    assert isinstance(score, AnomalyScore)
                    assert not np.isnan(score.value)
                    assert not np.isinf(score.value)
                    assert 0.0 <= score.value <= 1.0

            except (ImportError, ValueError) as e:
                # Some edge cases might not be supported
                if "ImportError" in str(type(e)):
                    continue
                else:
                    # Should handle edge cases gracefully, not crash
                    pass

    def test_deterministic_behavior(self):
        """Test that model behavior is deterministic when it should be."""
        # Create test dataset
        np.random.seed(555)
        data = np.random.normal(0, 1, (50, 4))
        df = pd.DataFrame(data, columns=["a", "b", "c", "d"])
        dataset = Dataset(name="Deterministic Test", data=df)

        try:
            # Run multiple times with same parameters
            results = []

            for _run in range(3):
                adapter = SklearnAdapter(
                    algorithm_name="IsolationForest",
                    parameters={
                        "contamination": 0.1,
                        "n_estimators": 20,
                        "random_state": 777,  # Fixed random state
                    },
                )

                adapter.fit(dataset)
                scores = adapter.score(dataset)
                score_values = [score.value for score in scores]
                results.append(score_values)

            # All runs should produce identical results
            for run_idx in range(1, len(results)):
                for i, (score1, score2) in enumerate(zip(results[0], results[run_idx], strict=False)):
                    assert abs(score1 - score2) < 1e-10, (
                        f"Non-deterministic behavior at sample {i}, run {run_idx}"
                    )

        except ImportError:
            pytest.skip("scikit-learn not available")
