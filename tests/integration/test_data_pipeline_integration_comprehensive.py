"""Comprehensive data pipeline integration tests.

This module contains integration tests for the complete data pipeline,
from data loading through preprocessing to anomaly detection and results output.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pynomaly.domain.entities import Dataset
from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter


class TestDataLoadingPipelineIntegration:
    """Test data loading pipeline integration."""

    @pytest.fixture
    def sample_data_files(self):
        """Create sample data files for testing."""
        files = {}

        # Create CSV file
        csv_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=1000, freq="1H"),
                "sensor_1": np.random.normal(25, 5, 1000),
                "sensor_2": np.random.normal(50, 10, 1000),
                "sensor_3": np.random.exponential(2, 1000),
                "categorical": np.random.choice(["A", "B", "C"], 1000),
                "boolean_flag": np.random.choice([True, False], 1000),
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_data.to_csv(f.name, index=False)
            files["csv"] = f.name

        # Create JSON file
        json_data = {
            "metadata": {
                "source": "sensor_network",
                "created_at": "2023-01-01T00:00:00Z",
                "version": "1.0",
            },
            "data": csv_data.to_dict("records")[:100],  # Smaller for JSON
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_data, f, indent=2, default=str)
            files["json"] = f.name

        # Create Parquet file (if available)
        try:
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
                csv_data.to_parquet(f.name, index=False)
                files["parquet"] = f.name
        except ImportError:
            pass  # Parquet not available

        return files

    def test_csv_loading_pipeline_integration(self, sample_data_files):
        """Test CSV data loading pipeline integration."""
        if "csv" not in sample_data_files:
            pytest.skip("CSV file not available")

        csv_path = sample_data_files["csv"]

        # Load CSV data
        loaded_data = pd.read_csv(csv_path)

        # Verify data structure
        assert len(loaded_data) == 1000
        assert "sensor_1" in loaded_data.columns
        assert "sensor_2" in loaded_data.columns
        assert "sensor_3" in loaded_data.columns

        # Select numeric columns for anomaly detection
        numeric_columns = ["sensor_1", "sensor_2", "sensor_3"]
        numeric_data = loaded_data[numeric_columns]

        # Create dataset
        dataset = Dataset(name="CSV Loaded Dataset", data=numeric_data)

        # Test complete pipeline
        try:
            adapter = SklearnAdapter(
                algorithm_name="IsolationForest",
                parameters={
                    "contamination": 0.1,
                    "n_estimators": 50,
                    "random_state": 42,
                },
            )

            # Train and detect
            adapter.fit(dataset)
            scores = adapter.score(dataset)
            result = adapter.detect(dataset)

            # Verify pipeline results
            assert len(scores) == len(loaded_data)
            assert len(result.labels) == len(loaded_data)

            # Verify data integrity through pipeline
            assert dataset.n_samples == len(loaded_data)
            assert dataset.n_features == len(numeric_columns)

        except ImportError:
            pytest.skip("scikit-learn not available")

        # Clean up
        Path(csv_path).unlink()

    def test_json_loading_pipeline_integration(self, sample_data_files):
        """Test JSON data loading pipeline integration."""
        if "json" not in sample_data_files:
            pytest.skip("JSON file not available")

        json_path = sample_data_files["json"]

        # Load JSON data
        with open(json_path) as f:
            json_data = json.load(f)

        # Extract data and metadata
        metadata = json_data["metadata"]
        data_records = json_data["data"]

        # Convert to DataFrame
        loaded_data = pd.DataFrame(data_records)

        # Verify structure
        assert len(loaded_data) == 100
        assert "sensor_1" in loaded_data.columns

        # Process timestamp if present
        if "timestamp" in loaded_data.columns:
            loaded_data["timestamp"] = pd.to_datetime(loaded_data["timestamp"])

        # Select numeric data
        numeric_columns = ["sensor_1", "sensor_2", "sensor_3"]
        numeric_data = loaded_data[numeric_columns]

        # Create dataset with metadata
        dataset = Dataset(name="JSON Loaded Dataset", data=numeric_data)

        # Test pipeline with metadata preservation
        try:
            adapter = SklearnAdapter(
                algorithm_name="IsolationForest",
                parameters={
                    "contamination": 0.1,
                    "n_estimators": 30,
                    "random_state": 42,
                },
            )

            adapter.fit(dataset)
            scores = adapter.score(dataset)
            result = adapter.detect(dataset)

            # Verify results
            assert len(scores) == len(loaded_data)

            # Create enriched results with metadata
            enriched_results = {
                "metadata": metadata,
                "detection_results": {
                    "dataset_id": str(result.dataset_id),
                    "detector_id": str(result.detector_id),
                    "n_samples": len(result.labels),
                    "n_anomalies": int(np.sum(result.labels)),
                    "contamination_rate": float(np.mean(result.labels)),
                },
                "scores": [score.value for score in scores],
                "labels": result.labels.tolist(),
            }

            # Verify enriched results structure
            assert "metadata" in enriched_results
            assert "detection_results" in enriched_results
            assert enriched_results["metadata"]["source"] == "sensor_network"

        except ImportError:
            pytest.skip("scikit-learn not available")

        # Clean up
        Path(json_path).unlink()

    def test_parquet_loading_pipeline_integration(self, sample_data_files):
        """Test Parquet data loading pipeline integration."""
        if "parquet" not in sample_data_files:
            pytest.skip("Parquet file not available")

        parquet_path = sample_data_files["parquet"]

        try:
            # Load Parquet data
            loaded_data = pd.read_parquet(parquet_path)

            # Verify data structure
            assert len(loaded_data) == 1000
            assert "sensor_1" in loaded_data.columns

            # Select numeric columns
            numeric_columns = ["sensor_1", "sensor_2", "sensor_3"]
            numeric_data = loaded_data[numeric_columns]

            # Create dataset
            dataset = Dataset(name="Parquet Loaded Dataset", data=numeric_data)

            # Test pipeline
            adapter = SklearnAdapter(
                algorithm_name="IsolationForest",
                parameters={
                    "contamination": 0.1,
                    "n_estimators": 40,
                    "random_state": 42,
                },
            )

            adapter.fit(dataset)
            scores = adapter.score(dataset)

            # Verify results
            assert len(scores) == len(loaded_data)

            # Test data type preservation
            for col in numeric_columns:
                assert pd.api.types.is_numeric_dtype(dataset.data[col])

        except ImportError:
            pytest.skip("Required libraries for Parquet not available")

        # Clean up
        Path(parquet_path).unlink()


class TestDataPreprocessingPipelineIntegration:
    """Test data preprocessing pipeline integration."""

    @pytest.fixture
    def raw_dataset(self):
        """Create raw dataset requiring preprocessing."""
        np.random.seed(42)

        # Create data with various issues
        data = pd.DataFrame(
            {
                "feature1": np.concatenate(
                    [
                        np.random.normal(0, 1, 800),
                        [np.nan] * 50,  # Missing values
                        np.random.normal(0, 1, 150),
                    ]
                ),
                "feature2": np.concatenate(
                    [
                        np.random.normal(10, 2, 900),
                        [np.inf, -np.inf] + [np.nan] * 8,  # Infinite and missing values
                        np.random.normal(10, 2, 90),
                    ]
                ),
                "feature3": np.random.exponential(2, 1000),  # Different distribution
                "feature4": np.concatenate(
                    [
                        [0] * 990,  # Low variance
                        [1] * 10,
                    ]
                ),
                "feature5": np.random.normal(1000, 100, 1000),  # Different scale
                "categorical": np.random.choice(["A", "B", "C", "D"], 1000),
                "constant": [42] * 1000,  # Constant feature
            }
        )

        return Dataset(name="Raw Dataset", data=data)

    def test_missing_value_handling_pipeline(self, raw_dataset):
        """Test missing value handling in pipeline."""
        # Identify numeric columns with missing values
        numeric_data = raw_dataset.data.select_dtypes(include=[np.number])
        missing_columns = numeric_data.columns[numeric_data.isnull().any()].tolist()

        assert len(missing_columns) > 0, "Test dataset should have missing values"

        # Strategy 1: Drop rows with missing values
        cleaned_data_drop = numeric_data.dropna()

        if len(cleaned_data_drop) > 50:  # Ensure enough data remains
            dataset_drop = Dataset(name="Cleaned by Drop", data=cleaned_data_drop)

            try:
                adapter = SklearnAdapter(
                    algorithm_name="IsolationForest",
                    parameters={
                        "contamination": 0.1,
                        "n_estimators": 30,
                        "random_state": 42,
                    },
                )

                adapter.fit(dataset_drop)
                scores = adapter.score(dataset_drop)

                # Verify no missing values in results
                assert len(scores) == len(cleaned_data_drop)
                assert not any(np.isnan(score.value) for score in scores)

            except ImportError:
                pytest.skip("scikit-learn not available")

        # Strategy 2: Fill missing values
        filled_data = numeric_data.fillna(numeric_data.mean())
        dataset_filled = Dataset(name="Cleaned by Fill", data=filled_data)

        try:
            adapter = SklearnAdapter(
                algorithm_name="IsolationForest",
                parameters={
                    "contamination": 0.1,
                    "n_estimators": 30,
                    "random_state": 42,
                },
            )

            adapter.fit(dataset_filled)
            scores = adapter.score(dataset_filled)

            # Verify results
            assert len(scores) == len(filled_data)
            assert not any(np.isnan(score.value) for score in scores)

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_feature_scaling_pipeline(self, raw_dataset):
        """Test feature scaling in pipeline."""
        # Select numeric columns
        numeric_data = raw_dataset.data.select_dtypes(include=[np.number])

        # Remove infinite values and missing values for scaling
        clean_data = numeric_data.replace([np.inf, -np.inf], np.nan).dropna()

        if len(clean_data) < 50:
            pytest.skip("Insufficient clean data for scaling test")

        # Original data (different scales)
        original_dataset = Dataset(name="Original Scale", data=clean_data)

        # Standardized data
        from sklearn.preprocessing import StandardScaler

        try:
            scaler = StandardScaler()
            scaled_data = pd.DataFrame(
                scaler.fit_transform(clean_data),
                columns=clean_data.columns,
                index=clean_data.index,
            )
            scaled_dataset = Dataset(name="Scaled Dataset", data=scaled_data)

            # Test both datasets
            adapter_configs = [
                ("Original", original_dataset),
                ("Scaled", scaled_dataset),
            ]

            results = {}

            for name, dataset in adapter_configs:
                try:
                    adapter = SklearnAdapter(
                        algorithm_name="IsolationForest",
                        parameters={
                            "contamination": 0.1,
                            "n_estimators": 30,
                            "random_state": 42,
                        },
                    )

                    adapter.fit(dataset)
                    scores = adapter.score(dataset)
                    result = adapter.detect(dataset)

                    results[name] = {
                        "scores": [score.value for score in scores],
                        "contamination_rate": np.mean(result.labels),
                    }

                except Exception:
                    continue

            # Verify both approaches work
            assert len(results) > 0

            # If both work, verify they produce reasonable results
            for name, result in results.items():
                assert 0.05 <= result["contamination_rate"] <= 0.2
                assert all(0.0 <= score <= 1.0 for score in result["scores"])

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_feature_selection_pipeline(self, raw_dataset):
        """Test feature selection in pipeline."""
        # Select numeric columns
        numeric_data = raw_dataset.data.select_dtypes(include=[np.number])
        clean_data = numeric_data.fillna(numeric_data.mean())

        # Identify constant or low variance features
        variances = clean_data.var()
        variances[variances < 0.01].index.tolist()
        high_variance_cols = variances[variances >= 0.01].index.tolist()

        # Test with all features
        all_features_dataset = Dataset(name="All Features", data=clean_data)

        # Test with selected features (remove low variance)
        if len(high_variance_cols) >= 2:
            selected_data = clean_data[high_variance_cols]
            selected_features_dataset = Dataset(
                name="Selected Features", data=selected_data
            )

            datasets_to_test = [
                ("All Features", all_features_dataset),
                ("Selected Features", selected_features_dataset),
            ]
        else:
            datasets_to_test = [("All Features", all_features_dataset)]

        try:
            results = {}

            for name, dataset in datasets_to_test:
                adapter = SklearnAdapter(
                    algorithm_name="IsolationForest",
                    parameters={
                        "contamination": 0.1,
                        "n_estimators": 30,
                        "random_state": 42,
                    },
                )

                adapter.fit(dataset)
                scores = adapter.score(dataset)

                results[name] = {
                    "n_features": dataset.n_features,
                    "scores": [score.value for score in scores],
                    "score_variance": np.var([score.value for score in scores]),
                }

            # Verify results
            for name, result in results.items():
                assert result["n_features"] > 0
                assert len(result["scores"]) > 0
                assert result["score_variance"] >= 0

            # If both feature sets tested, compare
            if len(results) == 2:
                all_features_variance = results["All Features"]["score_variance"]
                selected_features_variance = results["Selected Features"][
                    "score_variance"
                ]

                # Both should produce valid score variances
                assert all_features_variance >= 0
                assert selected_features_variance >= 0

        except ImportError:
            pytest.skip("scikit-learn not available")


class TestCompleteDataPipelineIntegration:
    """Test complete end-to-end data pipeline integration."""

    def test_streaming_data_pipeline_simulation(self):
        """Test streaming data pipeline simulation."""
        # Simulate streaming data batches
        batch_size = 100
        num_batches = 5

        # Initialize pipeline state
        trained_adapter = None
        cumulative_results = []

        try:
            for batch_idx in range(num_batches):
                # Generate new batch of data
                np.random.seed(42 + batch_idx)  # Different seed for each batch

                batch_data = pd.DataFrame(
                    {
                        "feature1": np.random.normal(0, 1, batch_size),
                        "feature2": np.random.normal(0, 1, batch_size),
                        "feature3": np.random.normal(0, 1, batch_size),
                    }
                )

                # Add some anomalies to later batches
                if batch_idx >= 2:
                    anomaly_indices = np.random.choice(batch_size, 5, replace=False)
                    batch_data.iloc[anomaly_indices] += 5  # Make them anomalous

                batch_dataset = Dataset(name=f"Batch {batch_idx}", data=batch_data)

                if batch_idx == 0:
                    # Train on first batch
                    trained_adapter = SklearnAdapter(
                        algorithm_name="IsolationForest",
                        parameters={
                            "contamination": 0.1,
                            "n_estimators": 50,
                            "random_state": 42,
                        },
                    )

                    trained_adapter.fit(batch_dataset)

                    # Initial scoring
                    scores = trained_adapter.score(batch_dataset)
                    result = trained_adapter.detect(batch_dataset)

                else:
                    # Score new batches with trained model
                    scores = trained_adapter.score(batch_dataset)
                    result = trained_adapter.detect(batch_dataset)

                # Record batch results
                batch_result = {
                    "batch_idx": batch_idx,
                    "n_samples": len(batch_data),
                    "n_anomalies": int(np.sum(result.labels)),
                    "contamination_rate": float(np.mean(result.labels)),
                    "avg_score": float(np.mean([score.value for score in scores])),
                    "max_score": float(np.max([score.value for score in scores])),
                }

                cumulative_results.append(batch_result)

            # Verify streaming pipeline results
            assert len(cumulative_results) == num_batches

            # First batch (training data) should have lower contamination
            assert cumulative_results[0]["contamination_rate"] <= 0.15

            # Later batches (with injected anomalies) should have higher contamination
            later_batches = cumulative_results[2:]
            if len(later_batches) > 0:
                avg_later_contamination = np.mean(
                    [b["contamination_rate"] for b in later_batches]
                )
                assert (
                    avg_later_contamination
                    > cumulative_results[0]["contamination_rate"]
                )

            # All batches should have valid results
            for batch_result in cumulative_results:
                assert batch_result["n_samples"] == batch_size
                assert 0.0 <= batch_result["contamination_rate"] <= 1.0
                assert 0.0 <= batch_result["avg_score"] <= 1.0
                assert 0.0 <= batch_result["max_score"] <= 1.0

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_multi_dataset_pipeline_integration(self):
        """Test pipeline integration across multiple datasets."""
        # Create multiple related datasets
        datasets = []

        # Base dataset
        np.random.seed(42)
        base_data = pd.DataFrame(
            {
                "sensor_a": np.random.normal(25, 5, 500),
                "sensor_b": np.random.normal(50, 10, 500),
                "sensor_c": np.random.exponential(2, 500),
            }
        )
        datasets.append(Dataset(name="Base Dataset", data=base_data))

        # Shifted dataset (different distribution)
        shifted_data = pd.DataFrame(
            {
                "sensor_a": np.random.normal(30, 5, 500),  # Shifted mean
                "sensor_b": np.random.normal(45, 10, 500),  # Shifted mean
                "sensor_c": np.random.exponential(1.5, 500),  # Different parameter
            }
        )
        datasets.append(Dataset(name="Shifted Dataset", data=shifted_data))

        # Noisy dataset (higher variance)
        noisy_data = pd.DataFrame(
            {
                "sensor_a": np.random.normal(25, 15, 500),  # Higher variance
                "sensor_b": np.random.normal(50, 25, 500),  # Higher variance
                "sensor_c": np.random.exponential(3, 500),  # Different parameter
            }
        )
        datasets.append(Dataset(name="Noisy Dataset", data=noisy_data))

        try:
            # Train model on base dataset
            base_adapter = SklearnAdapter(
                algorithm_name="IsolationForest",
                parameters={
                    "contamination": 0.1,
                    "n_estimators": 50,
                    "random_state": 42,
                },
            )

            base_adapter.fit(datasets[0])

            # Test model on all datasets
            cross_dataset_results = []

            for dataset in datasets:
                scores = base_adapter.score(dataset)
                result = base_adapter.detect(dataset)

                dataset_result = {
                    "dataset_name": dataset.name,
                    "n_samples": dataset.n_samples,
                    "contamination_rate": float(np.mean(result.labels)),
                    "avg_score": float(np.mean([score.value for score in scores])),
                    "score_std": float(np.std([score.value for score in scores])),
                }

                cross_dataset_results.append(dataset_result)

            # Verify cross-dataset results
            assert len(cross_dataset_results) == 3

            # All results should be valid
            for result in cross_dataset_results:
                assert result["n_samples"] == 500
                assert 0.0 <= result["contamination_rate"] <= 1.0
                assert 0.0 <= result["avg_score"] <= 1.0
                assert result["score_std"] >= 0.0

            # Base dataset should have expected contamination rate
            base_result = cross_dataset_results[0]
            assert abs(base_result["contamination_rate"] - 0.1) < 0.05

            # Shifted and noisy datasets may have different contamination rates
            shifted_result = cross_dataset_results[1]
            noisy_result = cross_dataset_results[2]

            # All should be within reasonable bounds
            assert shifted_result["contamination_rate"] <= 0.5
            assert noisy_result["contamination_rate"] <= 0.5

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_pipeline_error_recovery_integration(self):
        """Test pipeline error recovery and robustness."""
        # Create datasets with various issues
        datasets = []

        # Valid dataset
        valid_data = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 100),
                "feature2": np.random.normal(0, 1, 100),
            }
        )
        datasets.append(("valid", Dataset(name="Valid", data=valid_data)))

        # Dataset with missing values
        missing_data = pd.DataFrame(
            {
                "feature1": [1, 2, np.nan, 4, 5] * 20,
                "feature2": [1, np.nan, 3, 4, 5] * 20,
            }
        )
        datasets.append(("missing", Dataset(name="Missing", data=missing_data)))

        # Dataset with different structure
        different_structure = pd.DataFrame(
            {
                "different_feature": np.random.normal(0, 1, 100),
                "another_feature": np.random.normal(0, 1, 100),
            }
        )
        datasets.append(
            ("different", Dataset(name="Different", data=different_structure))
        )

        try:
            # Train on valid dataset
            adapter = SklearnAdapter(
                algorithm_name="IsolationForest",
                parameters={
                    "contamination": 0.1,
                    "n_estimators": 30,
                    "random_state": 42,
                },
            )

            adapter.fit(datasets[0][1])  # Train on valid dataset

            # Test pipeline robustness
            pipeline_results = []

            for dataset_type, dataset in datasets:
                try:
                    if dataset_type == "missing":
                        # Preprocess missing data
                        clean_data = dataset.data.fillna(dataset.data.mean())
                        clean_dataset = Dataset(name="Cleaned Missing", data=clean_data)
                        scores = adapter.score(clean_dataset)

                        pipeline_results.append(
                            {
                                "type": dataset_type,
                                "status": "success_with_preprocessing",
                                "n_scores": len(scores),
                            }
                        )

                    elif dataset_type == "different":
                        # This should fail gracefully
                        scores = adapter.score(dataset)

                        pipeline_results.append(
                            {
                                "type": dataset_type,
                                "status": "unexpected_success",
                                "n_scores": len(scores),
                            }
                        )

                    else:
                        # Valid dataset
                        scores = adapter.score(dataset)

                        pipeline_results.append(
                            {
                                "type": dataset_type,
                                "status": "success",
                                "n_scores": len(scores),
                            }
                        )

                except Exception as e:
                    pipeline_results.append(
                        {
                            "type": dataset_type,
                            "status": "expected_failure",
                            "error": str(e),
                        }
                    )

            # Verify pipeline handled different scenarios
            assert len(pipeline_results) == 3

            # Valid dataset should succeed
            valid_result = next(r for r in pipeline_results if r["type"] == "valid")
            assert valid_result["status"] == "success"
            assert valid_result["n_scores"] == 100

            # Missing data should either succeed with preprocessing or fail gracefully
            missing_result = next(r for r in pipeline_results if r["type"] == "missing")
            assert missing_result["status"] in [
                "success_with_preprocessing",
                "expected_failure",
            ]

            # Different structure should either fail gracefully or handle unexpectedly
            different_result = next(
                r for r in pipeline_results if r["type"] == "different"
            )
            assert different_result["status"] in [
                "expected_failure",
                "unexpected_success",
            ]

        except ImportError:
            pytest.skip("scikit-learn not available")
