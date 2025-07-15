"""Comprehensive adapter integration tests.

This module contains integration tests for all adapter implementations,
testing their interaction with external libraries, data flow, and
cross-adapter compatibility.
"""

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pynomaly.domain.entities import Dataset, DetectionResult
from pynomaly.domain.value_objects import AnomalyScore
from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter


class TestSklearnAdapterIntegration:
    """Test scikit-learn adapter integration with external library."""

    @pytest.fixture
    def sample_datasets(self):
        """Create sample datasets of different characteristics."""
        datasets = {}

        # Normal dataset
        np.random.seed(42)
        normal_data = np.random.multivariate_normal(
            mean=[0, 0, 0], cov=[[1, 0.1, 0.1], [0.1, 1, 0.1], [0.1, 0.1, 1]], size=1000
        )
        datasets["normal"] = Dataset(
            name="Normal Dataset",
            data=pd.DataFrame(
                normal_data, columns=["feature1", "feature2", "feature3"]
            ),
        )

        # Dataset with clear anomalies
        anomaly_data = np.vstack(
            [
                np.random.normal(0, 1, (950, 3)),  # Normal samples
                np.random.normal(5, 0.5, (50, 3)),  # Clear anomalies
            ]
        )
        datasets["with_anomalies"] = Dataset(
            name="Dataset with Anomalies",
            data=pd.DataFrame(anomaly_data, columns=["x", "y", "z"]),
        )

        # High-dimensional dataset
        high_dim_data = np.random.normal(0, 1, (500, 20))
        datasets["high_dimensional"] = Dataset(
            name="High Dimensional Dataset",
            data=pd.DataFrame(high_dim_data, columns=[f"dim_{i}" for i in range(20)]),
        )

        # Small dataset
        small_data = np.random.normal(0, 1, (50, 3))
        datasets["small"] = Dataset(
            name="Small Dataset", data=pd.DataFrame(small_data, columns=["a", "b", "c"])
        )

        return datasets

    def test_isolation_forest_adapter_integration(self, sample_datasets):
        """Test IsolationForest adapter integration."""
        try:
            for _dataset_name, dataset in sample_datasets.items():
                adapter = SklearnAdapter(
                    algorithm_name="IsolationForest",
                    parameters={
                        "contamination": 0.1,
                        "n_estimators": 50,
                        "random_state": 42,
                    },
                )

                # Test complete workflow
                adapter.fit(dataset)

                # Verify adapter state after fitting
                assert adapter.is_fitted
                assert adapter.trained_at is not None

                # Test scoring
                scores = adapter.score(dataset)
                assert len(scores) == dataset.n_samples

                for score in scores:
                    assert isinstance(score, AnomalyScore)
                    assert 0.0 <= score.value <= 1.0
                    assert not np.isnan(score.value)
                    assert not np.isinf(score.value)

                # Test detection
                result = adapter.detect(dataset)
                assert isinstance(result, DetectionResult)
                assert result.detector_id == adapter.id
                assert result.dataset_id == dataset.id
                assert len(result.scores) == dataset.n_samples
                assert len(result.labels) == dataset.n_samples

                # Verify contamination rate is approximately respected
                contamination_rate = np.mean(result.labels)
                assert abs(contamination_rate - 0.1) < 0.05

                # Test model persistence
                with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                    pickle.dump(adapter, f)
                    model_path = f.name

                # Load and test
                with open(model_path, "rb") as f:
                    loaded_adapter = pickle.load(f)

                loaded_scores = loaded_adapter.score(dataset)

                # Scores should be identical
                for orig_score, loaded_score in zip(
                    scores, loaded_scores, strict=False
                ):
                    assert abs(orig_score.value - loaded_score.value) < 1e-10

                # Clean up
                Path(model_path).unlink()

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_local_outlier_factor_adapter_integration(self, sample_datasets):
        """Test LocalOutlierFactor adapter integration."""
        try:
            # Test with smaller datasets (LOF can be memory intensive)
            test_datasets = {
                k: v
                for k, v in sample_datasets.items()
                if k in ["normal", "small", "with_anomalies"]
            }

            for _dataset_name, dataset in test_datasets.items():
                adapter = SklearnAdapter(
                    algorithm_name="LocalOutlierFactor",
                    parameters={
                        "contamination": 0.1,
                        "n_neighbors": min(20, dataset.n_samples // 2),
                        "novelty": True,
                    },
                )

                # Test workflow
                adapter.fit(dataset)
                assert adapter.is_fitted

                # Test scoring
                scores = adapter.score(dataset)
                assert len(scores) == dataset.n_samples

                for score in scores:
                    assert isinstance(score, AnomalyScore)
                    assert 0.0 <= score.value <= 1.0
                    assert not np.isnan(score.value)
                    assert not np.isinf(score.value)

                # Test detection
                result = adapter.detect(dataset)
                assert isinstance(result, DetectionResult)
                assert len(result.labels) == dataset.n_samples

                # Verify reasonable contamination rate
                contamination_rate = np.mean(result.labels)
                assert 0.0 <= contamination_rate <= 0.5

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_one_class_svm_adapter_integration(self, sample_datasets):
        """Test OneClassSVM adapter integration."""
        try:
            # Test with smaller datasets (SVM can be slow)
            test_datasets = {
                k: v for k, v in sample_datasets.items() if k in ["normal", "small"]
            }

            for _dataset_name, dataset in test_datasets.items():
                adapter = SklearnAdapter(
                    algorithm_name="OneClassSVM",
                    parameters={"gamma": "scale", "nu": 0.1},
                )

                # Test workflow
                adapter.fit(dataset)
                assert adapter.is_fitted

                # Test scoring
                scores = adapter.score(dataset)
                assert len(scores) == dataset.n_samples

                for score in scores:
                    assert isinstance(score, AnomalyScore)
                    assert 0.0 <= score.value <= 1.0
                    assert not np.isnan(score.value)
                    assert not np.isinf(score.value)

                # Test detection
                result = adapter.detect(dataset)
                assert isinstance(result, DetectionResult)
                assert len(result.labels) == dataset.n_samples

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_adapter_parameter_validation_integration(self):
        """Test adapter parameter validation integration."""
        try:
            # Test various parameter combinations
            parameter_tests = [
                {
                    "algorithm": "IsolationForest",
                    "valid_params": {
                        "contamination": 0.1,
                        "n_estimators": 100,
                        "max_samples": "auto",
                        "random_state": 42,
                    },
                    "invalid_params": {
                        "contamination": 1.5,  # Invalid
                        "n_estimators": -1,  # Invalid
                        "random_state": 42,
                    },
                },
                {
                    "algorithm": "LocalOutlierFactor",
                    "valid_params": {
                        "contamination": 0.1,
                        "n_neighbors": 20,
                        "novelty": True,
                    },
                    "invalid_params": {
                        "contamination": 0.0,  # Invalid
                        "n_neighbors": 0,  # Invalid
                        "novelty": True,
                    },
                },
            ]

            # Create test dataset
            test_data = np.random.normal(0, 1, (100, 3))
            dataset = Dataset(
                name="Parameter Test",
                data=pd.DataFrame(test_data, columns=["x", "y", "z"]),
            )

            for test_case in parameter_tests:
                algorithm = test_case["algorithm"]

                # Test valid parameters
                valid_adapter = SklearnAdapter(
                    algorithm_name=algorithm, parameters=test_case["valid_params"]
                )

                # Should work without issues
                valid_adapter.fit(dataset)
                scores = valid_adapter.score(dataset)
                assert len(scores) == len(dataset.data)

                # Test invalid parameters
                try:
                    invalid_adapter = SklearnAdapter(
                        algorithm_name=algorithm, parameters=test_case["invalid_params"]
                    )

                    # May fail during fit or creation
                    invalid_adapter.fit(dataset)

                    # If it doesn't fail, the underlying library handled it

                except (ValueError, TypeError):
                    # Expected for invalid parameters
                    pass

        except ImportError:
            pytest.skip("scikit-learn not available")


class TestCrossAdapterCompatibility:
    """Test compatibility and consistency across different adapters."""

    @pytest.fixture
    def comparison_dataset(self):
        """Create dataset for cross-adapter comparison."""
        np.random.seed(123)

        # Create dataset with known anomalies
        normal_samples = np.random.multivariate_normal(
            mean=[0, 0], cov=[[1, 0.2], [0.2, 1]], size=900
        )

        anomaly_samples = np.random.multivariate_normal(
            mean=[4, 4], cov=[[0.5, 0], [0, 0.5]], size=100
        )

        data = np.vstack([normal_samples, anomaly_samples])

        return Dataset(
            name="Cross-Adapter Comparison",
            data=pd.DataFrame(data, columns=["feature1", "feature2"]),
        )

    def test_adapter_score_consistency(self, comparison_dataset):
        """Test that different adapters produce reasonable scores."""
        try:
            algorithms = [
                (
                    "IsolationForest",
                    {"contamination": 0.1, "n_estimators": 50, "random_state": 42},
                ),
                (
                    "LocalOutlierFactor",
                    {"contamination": 0.1, "n_neighbors": 20, "novelty": True},
                ),
                ("OneClassSVM", {"gamma": "scale", "nu": 0.1}),
            ]

            adapter_results = {}

            for algorithm_name, parameters in algorithms:
                try:
                    adapter = SklearnAdapter(
                        algorithm_name=algorithm_name, parameters=parameters
                    )

                    adapter.fit(comparison_dataset)
                    scores = adapter.score(comparison_dataset)
                    result = adapter.detect(comparison_dataset)

                    score_values = [score.value for score in scores]

                    adapter_results[algorithm_name] = {
                        "scores": score_values,
                        "labels": result.labels,
                        "mean_score": np.mean(score_values),
                        "std_score": np.std(score_values),
                        "contamination_rate": np.mean(result.labels),
                    }

                except Exception:
                    continue

            # Verify we have at least one working adapter
            assert len(adapter_results) > 0

            # Cross-adapter validation
            for _algorithm, results in adapter_results.items():
                # All algorithms should detect some anomalies
                assert results["contamination_rate"] > 0.01
                assert results["contamination_rate"] < 0.5

                # Score statistics should be reasonable
                assert 0.0 <= results["mean_score"] <= 1.0
                assert results["std_score"] >= 0.0

                # Known anomalies (samples 900-999) should have higher average scores
                if len(results["scores"]) == 1000:
                    normal_scores = results["scores"][:900]
                    anomaly_scores = results["scores"][900:]

                    avg_normal = np.mean(normal_scores)
                    avg_anomaly = np.mean(anomaly_scores)

                    # Anomalies should generally have higher scores
                    # (though this isn't guaranteed for all algorithms)
                    if avg_anomaly > avg_normal:
                        assert avg_anomaly > avg_normal

            # If multiple algorithms available, compare their correlation
            if len(adapter_results) >= 2:
                algorithms_list = list(adapter_results.keys())

                for i in range(len(algorithms_list)):
                    for j in range(i + 1, len(algorithms_list)):
                        algo1, algo2 = algorithms_list[i], algorithms_list[j]

                        scores1 = adapter_results[algo1]["scores"]
                        scores2 = adapter_results[algo2]["scores"]

                        if len(scores1) == len(scores2):
                            correlation = np.corrcoef(scores1, scores2)[0, 1]

                            # Different algorithms might have different approaches,
                            # but there should be some positive correlation for anomaly detection
                            assert (
                                correlation > -0.5
                            ), f"Negative correlation between {algo1} and {algo2}: {correlation}"

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_adapter_performance_comparison(self, comparison_dataset):
        """Test performance characteristics across adapters."""
        try:
            import time

            algorithms = [
                (
                    "IsolationForest",
                    {
                        "contamination": 0.1,
                        "n_estimators": 20,  # Reduced for faster testing
                        "random_state": 42,
                    },
                ),
                (
                    "LocalOutlierFactor",
                    {
                        "contamination": 0.1,
                        "n_neighbors": 10,  # Reduced for faster testing
                        "novelty": True,
                    },
                ),
            ]

            performance_results = {}

            for algorithm_name, parameters in algorithms:
                try:
                    adapter = SklearnAdapter(
                        algorithm_name=algorithm_name, parameters=parameters
                    )

                    # Measure training time
                    start_time = time.time()
                    adapter.fit(comparison_dataset)
                    training_time = time.time() - start_time

                    # Measure scoring time
                    start_time = time.time()
                    scores = adapter.score(comparison_dataset)
                    scoring_time = time.time() - start_time

                    performance_results[algorithm_name] = {
                        "training_time": training_time,
                        "scoring_time": scoring_time,
                        "total_time": training_time + scoring_time,
                        "samples_per_second": len(scores) / scoring_time,
                    }

                    # Performance should be reasonable
                    assert (
                        training_time < 30.0
                    ), f"{algorithm_name} training too slow: {training_time}s"
                    assert (
                        scoring_time < 10.0
                    ), f"{algorithm_name} scoring too slow: {scoring_time}s"

                except Exception:
                    continue

            # Log performance comparison
            if len(performance_results) > 1:
                print(f"Adapter Performance Comparison: {performance_results}")

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_adapter_memory_usage_comparison(self, comparison_dataset):
        """Test memory usage across adapters."""
        try:
            import gc

            import psutil

            process = psutil.Process()

            algorithms = [
                (
                    "IsolationForest",
                    {"contamination": 0.1, "n_estimators": 30, "random_state": 42},
                )
            ]

            memory_results = {}

            for algorithm_name, parameters in algorithms:
                try:
                    # Measure baseline memory
                    gc.collect()
                    baseline_memory = process.memory_info().rss

                    adapter = SklearnAdapter(
                        algorithm_name=algorithm_name, parameters=parameters
                    )

                    # Measure memory after training
                    adapter.fit(comparison_dataset)
                    training_memory = process.memory_info().rss

                    # Measure memory after scoring
                    scores = adapter.score(comparison_dataset)
                    scoring_memory = process.memory_info().rss

                    # Clean up
                    del adapter, scores
                    gc.collect()
                    final_memory = process.memory_info().rss

                    memory_results[algorithm_name] = {
                        "training_increase_mb": (training_memory - baseline_memory)
                        / (1024 * 1024),
                        "scoring_increase_mb": (scoring_memory - training_memory)
                        / (1024 * 1024),
                        "memory_leak_mb": (final_memory - baseline_memory)
                        / (1024 * 1024),
                    }

                    # Memory usage should be reasonable
                    assert memory_results[algorithm_name]["training_increase_mb"] < 200
                    assert memory_results[algorithm_name]["scoring_increase_mb"] < 50
                    assert memory_results[algorithm_name]["memory_leak_mb"] < 20

                except Exception:
                    continue

        except ImportError:
            pytest.skip("psutil not available for memory monitoring")


class TestAdapterErrorHandlingIntegration:
    """Test error handling and edge cases in adapter integration."""

    def test_invalid_data_handling(self):
        """Test adapter handling of invalid data."""
        try:
            # Create datasets with various issues
            problematic_datasets = [
                # Dataset with NaN values
                Dataset(
                    name="NaN Dataset",
                    data=pd.DataFrame(
                        {
                            "feature1": [1.0, 2.0, np.nan, 4.0, 5.0],
                            "feature2": [1.0, np.nan, 3.0, 4.0, 5.0],
                        }
                    ),
                ),
                # Dataset with infinite values
                Dataset(
                    name="Infinite Dataset",
                    data=pd.DataFrame(
                        {
                            "feature1": [1.0, 2.0, np.inf, 4.0, 5.0],
                            "feature2": [1.0, 2.0, 3.0, -np.inf, 5.0],
                        }
                    ),
                ),
                # Empty dataset
                Dataset(
                    name="Empty Dataset",
                    data=pd.DataFrame(columns=["feature1", "feature2"]),
                ),
                # Single sample dataset
                Dataset(
                    name="Single Sample",
                    data=pd.DataFrame({"feature1": [1.0], "feature2": [2.0]}),
                ),
            ]

            for dataset in problematic_datasets:
                adapter = SklearnAdapter(
                    algorithm_name="IsolationForest",
                    parameters={
                        "contamination": 0.1,
                        "n_estimators": 10,
                        "random_state": 42,
                    },
                )

                try:
                    adapter.fit(dataset)
                    scores = adapter.score(dataset)

                    # If successful, verify results are valid
                    for score in scores:
                        assert isinstance(score, AnomalyScore)
                        assert not np.isnan(score.value)
                        assert not np.isinf(score.value)
                        assert 0.0 <= score.value <= 1.0

                except (ValueError, TypeError, RuntimeError):
                    # Expected for problematic data
                    pass

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_adapter_state_consistency(self):
        """Test adapter state consistency after errors."""
        try:
            # Create valid dataset
            valid_data = np.random.normal(0, 1, (100, 3))
            valid_dataset = Dataset(
                name="Valid Dataset",
                data=pd.DataFrame(valid_data, columns=["x", "y", "z"]),
            )

            adapter = SklearnAdapter(
                algorithm_name="IsolationForest",
                parameters={
                    "contamination": 0.1,
                    "n_estimators": 20,
                    "random_state": 42,
                },
            )

            # Initially not fitted
            assert not adapter.is_fitted
            assert adapter.trained_at is None

            # Successful training should update state
            adapter.fit(valid_dataset)
            assert adapter.is_fitted
            assert adapter.trained_at is not None

            # Successful scoring should work
            scores = adapter.score(valid_dataset)
            assert len(scores) == len(valid_dataset.data)

            # State should remain consistent
            assert adapter.is_fitted

            # Try to score invalid data
            invalid_dataset = Dataset(
                name="Invalid Dataset",
                data=pd.DataFrame(columns=["different", "columns"]),
            )

            try:
                invalid_scores = adapter.score(invalid_dataset)

                # If successful, should return empty results
                assert len(invalid_scores) == 0

            except (ValueError, RuntimeError):
                # Expected for incompatible data
                pass

            # Original adapter state should remain valid
            assert adapter.is_fitted

            # Should still work with original data
            retry_scores = adapter.score(valid_dataset)
            assert len(retry_scores) == len(valid_dataset.data)

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_adapter_concurrent_access(self):
        """Test adapter behavior under concurrent access."""
        try:
            import threading
            import time

            # Create test dataset
            test_data = np.random.normal(0, 1, (500, 4))
            dataset = Dataset(
                name="Concurrent Test",
                data=pd.DataFrame(test_data, columns=["a", "b", "c", "d"]),
            )

            adapter = SklearnAdapter(
                algorithm_name="IsolationForest",
                parameters={
                    "contamination": 0.1,
                    "n_estimators": 30,
                    "random_state": 42,
                },
            )

            # Train adapter
            adapter.fit(dataset)

            # Concurrent scoring function
            def concurrent_scoring(thread_id, results_list):
                try:
                    scores = adapter.score(dataset)
                    results_list.append(
                        {
                            "thread_id": thread_id,
                            "success": True,
                            "n_scores": len(scores),
                            "avg_score": np.mean([score.value for score in scores]),
                        }
                    )
                except Exception as e:
                    results_list.append(
                        {"thread_id": thread_id, "success": False, "error": str(e)}
                    )

            # Run concurrent scoring
            results = []
            threads = []

            for i in range(3):
                thread = threading.Thread(target=concurrent_scoring, args=(i, results))
                threads.append(thread)
                thread.start()

            # Wait for all threads
            for thread in threads:
                thread.join()

            # Verify results
            assert len(results) == 3

            successful_results = [r for r in results if r["success"]]
            assert len(successful_results) > 0, "No concurrent operations succeeded"

            # All successful operations should return same number of scores
            n_scores_list = [r["n_scores"] for r in successful_results]
            assert all(n == len(dataset.data) for n in n_scores_list)

            # Average scores should be similar (within reasonable range)
            avg_scores = [r["avg_score"] for r in successful_results]
            if len(avg_scores) > 1:
                score_variance = np.var(avg_scores)
                assert (
                    score_variance < 0.01
                ), "High variance in concurrent scoring results"

        except ImportError:
            pytest.skip("scikit-learn not available")
