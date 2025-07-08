"""Comprehensive performance regression tests.

This module contains regression tests to detect performance degradation
across versions and ensure that performance characteristics remain
within acceptable bounds.
"""

import json
import multiprocessing as mp
import tempfile
import time
import yaml
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pynomaly.domain.entities import Dataset
from pynomaly.domain.value_objects import AnomalyScore
from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter


def load_performance_config():
    """Load performance configuration from YAML file."""
    try:
        with open('scripts/performance/performance_config.yml', 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        return {}


def get_threshold(config, category, metric, default, algorithm='IsolationForest'):
    """Get threshold value from config or return default."""
    return config.get('algorithm_thresholds', {}).get(algorithm, {}).get(category, {}).get(metric, default)


def get_performance_threshold(config, category, metric, default):
    """Get performance threshold value from config or return default."""
    return config.get('performance_thresholds', {}).get(category, {}).get(metric, default)


class TestTrainingPerformanceRegression:
    """Test training performance regression across different scenarios."""

    @pytest.fixture
    def performance_datasets(self):
        """Create datasets of different sizes for performance testing."""
        datasets = {}

        # Small dataset
        np.random.seed(42)
        small_data = np.random.normal(0, 1, (1000, 5))
        datasets["small"] = Dataset(
            name="Small Dataset",
            data=pd.DataFrame(small_data, columns=[f"feature_{i}" for i in range(5)]),
        )

        # Medium dataset
        medium_data = np.random.normal(0, 1, (10000, 10))
        datasets["medium"] = Dataset(
            name="Medium Dataset",
            data=pd.DataFrame(medium_data, columns=[f"feature_{i}" for i in range(10)]),
        )

        # Large dataset
        large_data = np.random.normal(0, 1, (50000, 5))
        datasets["large"] = Dataset(
            name="Large Dataset",
            data=pd.DataFrame(large_data, columns=[f"feature_{i}" for i in range(5)]),
        )

        # Wide dataset (many features)
        wide_data = np.random.normal(0, 1, (5000, 50))
        datasets["wide"] = Dataset(
            name="Wide Dataset",
            data=pd.DataFrame(wide_data, columns=[f"feature_{i}" for i in range(50)]),
        )

        return datasets

    def test_isolation_forest_training_performance(self, performance_datasets):
        """Test IsolationForest training performance regression."""
        performance_results = {}
        config = load_performance_config()

        for dataset_name, dataset in performance_datasets.items():
            try:
                adapter = SklearnAdapter(
                    algorithm_name="IsolationForest",
                    parameters={
                        "contamination": 0.1,
                        "n_estimators": 100,
                        "random_state": 42,
                        "n_jobs": 1,  # Single thread for consistent timing
                    },
                )

                # Measure training time
                start_time = time.time()
                adapter.fit(dataset)
                training_time = time.time() - start_time

                performance_results[dataset_name] = {
                    "training_time": training_time,
                    "n_samples": dataset.n_samples,
                    "n_features": dataset.n_features,
                    "samples_per_second": dataset.n_samples / training_time,
                }

                # Performance thresholds based on dataset size
                default_time_threshold = {'small': 5.0, 'medium': 30.0, 'large': 120.0, 'wide': 60.0}
                time_threshold = get_threshold(config, 'execution_time', 'max_execution_time_seconds', default_time_threshold[dataset_name])
                assert (
                    training_time < time_threshold
                ), f"{dataset_name.capitalize()} dataset training too slow: {training_time}s"

                # Throughput should be reasonable
                min_throughput = get_performance_threshold(config, 'throughput', 'min_throughput_samples_per_second', 100)
                samples_per_second = dataset.n_samples / training_time
                assert (
                    samples_per_second > min_throughput
                ), f"Training throughput too low: {samples_per_second} samples/s"

            except ImportError:
                pytest.skip("scikit-learn not available")

        # Log performance results for monitoring
        print(
            f"Training performance results: {json.dumps(performance_results, indent=2)}"
        )

    def test_local_outlier_factor_training_performance(self, performance_datasets):
        """Test LocalOutlierFactor training performance regression."""
        # LOF is typically slower, so we use smaller datasets
        test_datasets = {
            k: v for k, v in performance_datasets.items() if k in ["small", "medium"]
        }
        config = load_performance_config()

        for dataset_name, dataset in test_datasets.items():
            try:
                adapter = SklearnAdapter(
                    algorithm_name="LocalOutlierFactor",
                    parameters={
                        "contamination": 0.1,
                        "n_neighbors": 20,
                        "novelty": True,
                        "n_jobs": 1,
                    },
                )

                start_time = time.time()
                adapter.fit(dataset)
                training_time = time.time() - start_time

                # LOF performance thresholds (more lenient)
                default_time_threshold = {'small': 10.0, 'medium': 60.0}
                time_threshold = get_threshold(config, 'execution_time', 'max_execution_time_seconds', default_time_threshold[dataset_name], 'LocalOutlierFactor')
                assert (
                    training_time < time_threshold
                ), f"LOF {dataset_name} dataset training too slow: {training_time}s"

            except ImportError:
                continue

    def test_training_scalability_regression(self, performance_datasets):
        """Test training time scalability with dataset size."""
        scalability_results = []

        # Test with increasing dataset sizes
        test_sizes = [1000, 5000, 10000, 25000]

        for size in test_sizes:
            # Create dataset of specific size
            np.random.seed(42)
            data = np.random.normal(0, 1, (size, 5))
            dataset = Dataset(
                name=f"Scalability Test {size}",
                data=pd.DataFrame(data, columns=[f"f_{i}" for i in range(5)]),
            )

            try:
                adapter = SklearnAdapter(
                    algorithm_name="IsolationForest",
                    parameters={
                        "contamination": 0.1,
                        "n_estimators": 50,  # Fewer estimators for faster testing
                        "random_state": 42,
                        "n_jobs": 1,
                    },
                )

                start_time = time.time()
                adapter.fit(dataset)
                training_time = time.time() - start_time

                scalability_results.append(
                    {
                        "size": size,
                        "time": training_time,
                        "time_per_sample": training_time / size,
                    }
                )

            except ImportError:
                continue

        if len(scalability_results) >= 2:
            # Check that scaling is reasonable (not exponential)
            for i in range(1, len(scalability_results)):
                prev_result = scalability_results[i - 1]
                curr_result = scalability_results[i]

                size_ratio = curr_result["size"] / prev_result["size"]
                time_ratio = curr_result["time"] / prev_result["time"]

                # Time should not increase faster than O(n^2)
                max_acceptable_ratio = size_ratio**2
                assert (
                    time_ratio <= max_acceptable_ratio
                ), f"Poor time scalability: {time_ratio}x time for {size_ratio}x data"


class TestInferencePerformanceRegression:
    """Test inference/scoring performance regression."""

    @pytest.fixture
    def trained_models(self, performance_datasets):
        """Create pre-trained models for inference testing."""
        trained_models = {}

        # Train models on small dataset for fast setup
        dataset = performance_datasets["small"]

        try:
            # IsolationForest
            if_adapter = SklearnAdapter(
                algorithm_name="IsolationForest",
                parameters={
                    "contamination": 0.1,
                    "n_estimators": 100,
                    "random_state": 42,
                },
            )
            if_adapter.fit(dataset)
            trained_models["isolation_forest"] = if_adapter

            # LocalOutlierFactor
            lof_adapter = SklearnAdapter(
                algorithm_name="LocalOutlierFactor",
                parameters={"contamination": 0.1, "n_neighbors": 20, "novelty": True},
            )
            lof_adapter.fit(dataset)
            trained_models["local_outlier_factor"] = lof_adapter

        except ImportError:
            pass

        return trained_models

    def test_single_sample_inference_performance(
        self, trained_models, performance_datasets
    ):
        """Test single sample inference performance."""
        # Create single sample datasets
        single_sample_data = performance_datasets["small"].data.iloc[:1]
        single_sample_dataset = Dataset(name="Single Sample", data=single_sample_data)
        config = load_performance_config()

        for model_name, model in trained_models.items():
            # Measure inference time for single sample
            start_time = time.time()
            scores = model.score(single_sample_dataset)
            inference_time = time.time() - start_time

            # Single sample inference should be very fast
            single_sample_threshold = get_performance_threshold(config, 'execution_time', 'single_sample_max_seconds', 1.0)
            assert (
                inference_time < single_sample_threshold
            ), f"{model_name} single sample inference too slow: {inference_time}s"
            assert len(scores) == 1
            assert isinstance(scores[0], AnomalyScore)

    def test_batch_inference_performance(self, trained_models, performance_datasets):
        """Test batch inference performance."""
        batch_sizes = [100, 1000, 5000]
        config = load_performance_config()

        for model_name, model in trained_models.items():
            for batch_size in batch_sizes:
                # Create batch dataset
                if batch_size <= len(performance_datasets["medium"].data):
                    batch_data = performance_datasets["medium"].data.iloc[:batch_size]
                    batch_dataset = Dataset(name=f"Batch {batch_size}", data=batch_data)

                    # Measure batch inference time
                    start_time = time.time()
                    scores = model.score(batch_dataset)
                    inference_time = time.time() - start_time

                    # Calculate throughput
                    throughput = batch_size / inference_time

                    # Performance thresholds
                    default_thresholds = {100: 2.0, 1000: 10.0, 5000: 30.0}
                    time_threshold = get_performance_threshold(config, 'execution_time', f'batch_{batch_size}_max_seconds', default_thresholds[batch_size])
                    assert (
                        inference_time < time_threshold
                    ), f"{model_name} batch {batch_size} inference too slow: {inference_time}s"

                    # Throughput should be reasonable
                    min_throughput = get_performance_threshold(config, 'throughput', 'min_throughput_samples_per_second', 50)
                    assert (
                        throughput > min_throughput
                    ), f"{model_name} throughput too low: {throughput} samples/s"

                    # Verify results
                    assert len(scores) == batch_size

    def test_inference_scalability_regression(self, trained_models):
        """Test inference time scalability."""
        if not trained_models:
            pytest.skip("No trained models available")

        model = list(trained_models.values())[0]  # Use first available model

        # Test with increasing batch sizes
        test_sizes = [100, 500, 1000, 2500]
        inference_results = []

        for size in test_sizes:
            # Create test dataset
            np.random.seed(42)
            data = np.random.normal(0, 1, (size, 5))
            dataset = Dataset(
                name=f"Inference Test {size}",
                data=pd.DataFrame(data, columns=[f"f_{i}" for i in range(5)]),
            )

            # Measure inference time
            start_time = time.time()
            scores = model.score(dataset)
            inference_time = time.time() - start_time

            inference_results.append(
                {
                    "size": size,
                    "time": inference_time,
                    "throughput": size / inference_time,
                }
            )

            assert len(scores) == size

        # Check scalability
        if len(inference_results) >= 2:
            for i in range(1, len(inference_results)):
                prev_result = inference_results[i - 1]
                curr_result = inference_results[i]

                size_ratio = curr_result["size"] / prev_result["size"]
                time_ratio = curr_result["time"] / prev_result["time"]

                # Inference time should scale roughly linearly
                max_acceptable_ratio = size_ratio * 1.5  # Allow 50% overhead
                assert (
                    time_ratio <= max_acceptable_ratio
                ), f"Poor inference scalability: {time_ratio}x time for {size_ratio}x data"


class TestMemoryPerformanceRegression:
    """Test memory usage performance regression."""

    def test_training_memory_usage(self, performance_datasets):
        """Test memory usage during training."""
        try:
            import psutil

            process = psutil.Process()
        except ImportError:
            pytest.skip("psutil not available for memory monitoring")

        import gc

        for dataset_name, dataset in performance_datasets.items():
            if dataset_name == "large":
                continue  # Skip large dataset to avoid memory issues

            try:
                # Measure baseline memory
                gc.collect()
                baseline_memory = process.memory_info().rss

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

                # Measure memory after training
                scores = adapter.score(dataset)
                scoring_memory = process.memory_info().rss

                # Clean up
                del adapter, scores
                gc.collect()
                final_memory = process.memory_info().rss

                # Calculate memory usage
                training_increase = training_memory - baseline_memory
                scoring_increase = scoring_memory - training_memory
                memory_leak = final_memory - baseline_memory

                # Memory usage thresholds (in MB)
                training_mb = training_increase / (1024 * 1024)
                scoring_mb = scoring_increase / (1024 * 1024)
                leak_mb = memory_leak / (1024 * 1024)

                # Memory usage should be reasonable
                config = load_performance_config()
                default_memory_thresholds = {'small': 100, 'medium': 500, 'wide': 200}
                memory_threshold = config.get('performance_thresholds', {}).get('memory_usage', {}).get(f'{dataset_name}_max_memory_mb', default_memory_thresholds[dataset_name])
                assert (
                    training_mb < memory_threshold
                ), f"{dataset_name.capitalize()} dataset training uses too much memory: {training_mb} MB"

                # Scoring should not significantly increase memory
                scoring_threshold = config.get('performance_thresholds', {}).get('memory_usage', {}).get('scoring_max_memory_mb', 50)
                assert (
                    scoring_mb < scoring_threshold
                ), f"Scoring increases memory too much: {scoring_mb} MB"

                # Memory leak should be minimal
                leak_threshold = config.get('performance_thresholds', {}).get('memory_usage', {}).get('leak_max_memory_mb', 20)
                assert leak_mb < leak_threshold, f"Memory leak detected: {leak_mb} MB"

            except ImportError:
                continue

    def test_concurrent_memory_usage(self):
        """Test memory usage under concurrent operations."""
        try:
            import psutil

            process = psutil.Process()
        except ImportError:
            pytest.skip("psutil not available")

        import gc

        # Measure baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss

        def train_model(seed):
            """Train a model in a thread."""
            np.random.seed(seed)
            data = np.random.normal(0, 1, (1000, 5))
            dataset = Dataset(
                name=f"Concurrent {seed}",
                data=pd.DataFrame(data, columns=[f"f_{i}" for i in range(5)]),
            )

            try:
                adapter = SklearnAdapter(
                    algorithm_name="IsolationForest",
                    parameters={
                        "contamination": 0.1,
                        "n_estimators": 20,
                        "random_state": seed,
                    },
                )

                adapter.fit(dataset)
                scores = adapter.score(dataset)
                return len(scores)

            except ImportError:
                return 0

        # Run concurrent training
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(train_model, i) for i in range(3)]
            results = [future.result() for future in futures]

        # Measure memory after concurrent operations
        concurrent_memory = process.memory_info().rss

        # Clean up
        gc.collect()
        final_memory = process.memory_info().rss

        # Calculate memory usage
        concurrent_increase = concurrent_memory - baseline_memory
        memory_leak = final_memory - baseline_memory

        concurrent_mb = concurrent_increase / (1024 * 1024)
        leak_mb = memory_leak / (1024 * 1024)

        # Verify operations completed
        assert all(result > 0 for result in results if result is not None)

        # Concurrent memory usage should not be excessive
        config = load_performance_config()
        concurrent_threshold = config.get('performance_thresholds', {}).get('memory_usage', {}).get('concurrent_max_memory_mb', 300)
        assert (
            concurrent_mb < concurrent_threshold
        ), f"Concurrent operations use too much memory: {concurrent_mb} MB"

        # Memory leak should be minimal
        concurrent_leak_threshold = config.get('performance_thresholds', {}).get('memory_usage', {}).get('concurrent_leak_max_memory_mb', 50)
        assert leak_mb < concurrent_leak_threshold, f"Concurrent operations cause memory leak: {leak_mb} MB"


class TestConcurrencyPerformanceRegression:
    """Test concurrency performance regression."""

    def test_thread_safety_performance(self):
        """Test thread safety doesn't degrade performance."""
        # Create test dataset
        np.random.seed(42)
        data = np.random.normal(0, 1, (1000, 5))
        dataset = Dataset(
            name="Thread Safety Test",
            data=pd.DataFrame(data, columns=[f"f_{i}" for i in range(5)]),
        )

        try:
            # Train model once
            adapter = SklearnAdapter(
                algorithm_name="IsolationForest",
                parameters={
                    "contamination": 0.1,
                    "n_estimators": 50,
                    "random_state": 42,
                },
            )
            adapter.fit(dataset)

            # Test sequential inference
            sequential_times = []
            for _ in range(5):
                start_time = time.time()
                adapter.score(dataset)
                sequential_times.append(time.time() - start_time)

            avg_sequential_time = np.mean(sequential_times)

            # Test concurrent inference
            def concurrent_inference():
                start_time = time.time()
                adapter.score(dataset)
                return time.time() - start_time

            concurrent_times = []
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(concurrent_inference) for _ in range(3)]
                concurrent_times = [future.result() for future in futures]

            avg_concurrent_time = np.mean(concurrent_times)

            # Concurrent inference should not be significantly slower
            config = load_performance_config()
            max_slowdown_ratio = config.get('performance_thresholds', {}).get('concurrency', {}).get('max_slowdown_ratio', 3.0)
            slowdown_ratio = avg_concurrent_time / avg_sequential_time
            assert (
                slowdown_ratio < max_slowdown_ratio
            ), f"Concurrent inference too slow: {slowdown_ratio}x slower"

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_multiprocessing_performance(self):
        """Test multiprocessing performance characteristics."""
        if mp.cpu_count() < 2:
            pytest.skip("Multiprocessing test requires multiple CPU cores")

        def train_and_score(seed):
            """Train and score in a separate process."""
            np.random.seed(seed)
            data = np.random.normal(0, 1, (2000, 5))
            dataset = Dataset(
                name=f"MP Test {seed}",
                data=pd.DataFrame(data, columns=[f"f_{i}" for i in range(5)]),
            )

            try:
                adapter = SklearnAdapter(
                    algorithm_name="IsolationForest",
                    parameters={
                        "contamination": 0.1,
                        "n_estimators": 30,
                        "random_state": seed,
                    },
                )

                start_time = time.time()
                adapter.fit(dataset)
                scores = adapter.score(dataset)
                total_time = time.time() - start_time

                return {"seed": seed, "time": total_time, "n_scores": len(scores)}

            except ImportError:
                return {"seed": seed, "time": float("inf"), "n_scores": 0}

        # Test sequential processing
        start_time = time.time()
        sequential_results = [train_and_score(i) for i in range(2)]
        sequential_time = time.time() - start_time

        # Test parallel processing
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=2) as executor:
            parallel_results = list(executor.map(train_and_score, range(2)))
        parallel_time = time.time() - start_time

        # Verify results
        assert all(result["n_scores"] > 0 for result in sequential_results)
        assert all(result["n_scores"] > 0 for result in parallel_results)

        # Parallel processing should provide some speedup
        config = load_performance_config()
        min_speedup = config.get('performance_thresholds', {}).get('concurrency', {}).get('min_multiprocessing_speedup', 0.8)
        speedup = sequential_time / parallel_time
        assert speedup > min_speedup, f"Multiprocessing provides no benefit: {speedup}x speedup"

        # Should not be slower than sequential (accounting for overhead)
        max_overhead_ratio = config.get('performance_thresholds', {}).get('concurrency', {}).get('max_multiprocessing_overhead_ratio', 1.5)
        assert (
            parallel_time < sequential_time * max_overhead_ratio
        ), "Multiprocessing significantly slower than sequential"


class TestIOPerformanceRegression:
    """Test I/O performance regression."""

    def test_model_serialization_performance(self):
        """Test model serialization/deserialization performance."""
        # Create and train model
        np.random.seed(42)
        data = np.random.normal(0, 1, (5000, 10))
        dataset = Dataset(
            name="Serialization Test",
            data=pd.DataFrame(data, columns=[f"f_{i}" for i in range(10)]),
        )

        try:
            adapter = SklearnAdapter(
                algorithm_name="IsolationForest",
                parameters={
                    "contamination": 0.1,
                    "n_estimators": 100,
                    "random_state": 42,
                },
            )
            adapter.fit(dataset)

            # Test pickle serialization performance
            import pickle

            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                start_time = time.time()
                pickle.dump(adapter, f)
                serialization_time = time.time() - start_time
                pickle_path = f.name

            # Test pickle deserialization performance
            start_time = time.time()
            with open(pickle_path, "rb") as f:
                loaded_adapter = pickle.load(f)
            deserialization_time = time.time() - start_time

            # Test loaded model works
            scores = loaded_adapter.score(dataset)
            assert len(scores) == len(dataset.data)

            # Performance thresholds
            config = load_performance_config()
            serialize_threshold = config.get('performance_thresholds', {}).get('io', {}).get('serialization_max_seconds', 5.0)
            deserialize_threshold = config.get('performance_thresholds', {}).get('io', {}).get('deserialization_max_seconds', 3.0)
            assert (
                serialization_time < serialize_threshold
            ), f"Model serialization too slow: {serialization_time}s"
            assert (
                deserialization_time < deserialize_threshold
            ), f"Model deserialization too slow: {deserialization_time}s"

            # Clean up
            Path(pickle_path).unlink()

            # Test joblib if available
            try:
                import joblib

                with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
                    start_time = time.time()
                    joblib.dump(adapter, f.name)
                    joblib_serialize_time = time.time() - start_time
                    joblib_path = f.name

                start_time = time.time()
                joblib.load(joblib_path)
                joblib_deserialize_time = time.time() - start_time

                # Joblib should be reasonably fast
                joblib_serialize_threshold = config.get('performance_thresholds', {}).get('io', {}).get('joblib_serialization_max_seconds', 5.0)
                joblib_deserialize_threshold = config.get('performance_thresholds', {}).get('io', {}).get('joblib_deserialization_max_seconds', 3.0)
                assert (
                    joblib_serialize_time < joblib_serialize_threshold
                ), f"Joblib serialization too slow: {joblib_serialize_time}s"
                assert (
                    joblib_deserialize_time < joblib_deserialize_threshold
                ), f"Joblib deserialization too slow: {joblib_deserialize_time}s"

                # Clean up
                Path(joblib_path).unlink()

            except ImportError:
                pass  # joblib not available

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_data_loading_performance(self):
        """Test data loading performance."""
        # Create test data files
        data_sizes = [1000, 10000, 50000]

        for size in data_sizes:
            # Generate test data
            np.random.seed(42)
            data = pd.DataFrame(
                {
                    "feature1": np.random.normal(0, 1, size),
                    "feature2": np.random.exponential(1, size),
                    "feature3": np.random.uniform(-1, 1, size),
                    "category": np.random.choice(["A", "B", "C"], size),
                }
            )

            # Test CSV loading performance
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
                csv_path = f.name

            # Write CSV
            start_time = time.time()
            data.to_csv(csv_path, index=False)
            csv_write_time = time.time() - start_time

            # Read CSV
            start_time = time.time()
            loaded_csv = pd.read_csv(csv_path)
            csv_read_time = time.time() - start_time

            # Verify data integrity
            assert len(loaded_csv) == size
            assert len(loaded_csv.columns) == 4

            # Performance thresholds based on size
            config = load_performance_config()
            default_write_thresholds = {1000: 2.0, 10000: 5.0, 50000: 15.0}
            default_read_thresholds = {1000: 1.0, 10000: 3.0, 50000: 10.0}
            
            size_key = min(default_write_thresholds.keys(), key=lambda x: abs(x - size))
            write_threshold = config.get('performance_thresholds', {}).get('io', {}).get(f'csv_write_{size_key}_max_seconds', default_write_thresholds[size_key])
            read_threshold = config.get('performance_thresholds', {}).get('io', {}).get(f'csv_read_{size_key}_max_seconds', default_read_thresholds[size_key])
            
            assert (
                csv_write_time < write_threshold
            ), f"CSV write too slow for {size} rows: {csv_write_time}s"
            assert (
                csv_read_time < read_threshold
            ), f"CSV read too slow for {size} rows: {csv_read_time}s"

            # Clean up
            Path(csv_path).unlink()

            # Test Parquet if available
            try:
                with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
                    parquet_path = f.name

                start_time = time.time()
                data.to_parquet(parquet_path, index=False)
                time.time() - start_time

                start_time = time.time()
                pd.read_parquet(parquet_path)
                parquet_read_time = time.time() - start_time

                # Parquet should be faster than CSV for larger datasets
                if size >= 10000:
                    assert (
                        parquet_read_time <= csv_read_time
                    ), "Parquet not faster than CSV for large data"

                # Clean up
                Path(parquet_path).unlink()

            except ImportError:
                pass  # Parquet not available
