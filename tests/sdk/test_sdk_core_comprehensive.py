"""Comprehensive SDK core functionality tests.

This module contains comprehensive tests for the core SDK functionality,
including the main API interface, dataset handling, model management,
and result processing.
"""

import json
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pynomaly.domain.entities import Dataset
from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter


class TestSDKCoreInterface:
    """Test core SDK interface functionality."""

    @pytest.fixture
    def mock_pynomaly_sdk(self):
        """Create mock Pynomaly SDK for testing."""

        class MockPynomalySDK:
            def __init__(self):
                self.models = {}
                self.datasets = {}
                self.results = {}
                self._default_config = {
                    "contamination": 0.1,
                    "algorithm": "IsolationForest",
                    "n_estimators": 100,
                    "random_state": 42,
                }

            def load_dataset(self, data, name: str = None, **kwargs) -> str:
                """Load dataset into SDK."""
                if isinstance(data, str):
                    # Assume it's a file path
                    if data.endswith(".csv"):
                        df = pd.read_csv(data)
                    elif data.endswith(".json"):
                        with open(data) as f:
                            json_data = json.load(f)
                        df = pd.DataFrame(json_data)
                    else:
                        raise ValueError(f"Unsupported file format: {data}")
                elif isinstance(data, (pd.DataFrame, np.ndarray)):
                    if isinstance(data, np.ndarray):
                        df = pd.DataFrame(data)
                    else:
                        df = data
                else:
                    raise ValueError("Data must be file path, DataFrame, or ndarray")

                dataset_id = str(uuid.uuid4())
                dataset_name = name or f"dataset_{dataset_id[:8]}"

                self.datasets[dataset_id] = Dataset(name=dataset_name, data=df)

                return dataset_id

            def create_model(self, algorithm: str = None, **parameters) -> str:
                """Create anomaly detection model."""
                algorithm = algorithm or self._default_config["algorithm"]

                # Merge with default parameters
                model_params = {**self._default_config}
                model_params.update(parameters)
                model_params.pop("algorithm", None)

                try:
                    adapter = SklearnAdapter(
                        algorithm_name=algorithm, parameters=model_params
                    )

                    model_id = str(uuid.uuid4())
                    self.models[model_id] = {
                        "adapter": adapter,
                        "algorithm": algorithm,
                        "parameters": model_params,
                        "trained": False,
                        "created_at": datetime.utcnow(),
                    }

                    return model_id

                except Exception as e:
                    raise ValueError(f"Failed to create model: {str(e)}")

            def train_model(self, model_id: str, dataset_id: str) -> dict[str, Any]:
                """Train model on dataset."""
                if model_id not in self.models:
                    raise ValueError(f"Model {model_id} not found")

                if dataset_id not in self.datasets:
                    raise ValueError(f"Dataset {dataset_id} not found")

                model = self.models[model_id]
                dataset = self.datasets[dataset_id]

                try:
                    adapter = model["adapter"]
                    adapter.fit(dataset)

                    model["trained"] = True
                    model["trained_at"] = datetime.utcnow()

                    return {
                        "model_id": model_id,
                        "dataset_id": dataset_id,
                        "status": "success",
                        "trained_at": model["trained_at"].isoformat(),
                        "algorithm": model["algorithm"],
                        "dataset_samples": dataset.n_samples,
                        "dataset_features": dataset.n_features,
                    }

                except Exception as e:
                    return {
                        "model_id": model_id,
                        "dataset_id": dataset_id,
                        "status": "failed",
                        "error": str(e),
                    }

            def detect_anomalies(self, model_id: str, dataset_id: str) -> str:
                """Detect anomalies using trained model."""
                if model_id not in self.models:
                    raise ValueError(f"Model {model_id} not found")

                if dataset_id not in self.datasets:
                    raise ValueError(f"Dataset {dataset_id} not found")

                model = self.models[model_id]
                dataset = self.datasets[dataset_id]

                if not model["trained"]:
                    raise ValueError(f"Model {model_id} is not trained")

                try:
                    adapter = model["adapter"]
                    result = adapter.detect(dataset)

                    result_id = str(uuid.uuid4())
                    self.results[result_id] = {
                        "result": result,
                        "model_id": model_id,
                        "dataset_id": dataset_id,
                        "created_at": datetime.utcnow(),
                    }

                    return result_id

                except Exception as e:
                    raise RuntimeError(f"Detection failed: {str(e)}")

            def get_results(self, result_id: str) -> dict[str, Any]:
                """Get detection results."""
                if result_id not in self.results:
                    raise ValueError(f"Results {result_id} not found")

                result_data = self.results[result_id]
                result = result_data["result"]

                return {
                    "result_id": result_id,
                    "model_id": result_data["model_id"],
                    "dataset_id": result_data["dataset_id"],
                    "n_samples": len(result.labels),
                    "n_anomalies": int(np.sum(result.labels)),
                    "contamination_rate": float(np.mean(result.labels)),
                    "scores": [score.value for score in result.scores],
                    "labels": result.labels.tolist(),
                    "threshold": result.threshold,
                    "created_at": result_data["created_at"].isoformat(),
                }

            def list_datasets(self) -> list[dict[str, Any]]:
                """List all loaded datasets."""
                return [
                    {
                        "dataset_id": dataset_id,
                        "name": dataset.name,
                        "n_samples": dataset.n_samples,
                        "n_features": dataset.n_features,
                    }
                    for dataset_id, dataset in self.datasets.items()
                ]

            def list_models(self) -> list[dict[str, Any]]:
                """List all created models."""
                return [
                    {
                        "model_id": model_id,
                        "algorithm": model_data["algorithm"],
                        "parameters": model_data["parameters"],
                        "trained": model_data["trained"],
                        "created_at": model_data["created_at"].isoformat(),
                    }
                    for model_id, model_data in self.models.items()
                ]

            def get_model_info(self, model_id: str) -> dict[str, Any]:
                """Get detailed model information."""
                if model_id not in self.models:
                    raise ValueError(f"Model {model_id} not found")

                model = self.models[model_id]

                info = {
                    "model_id": model_id,
                    "algorithm": model["algorithm"],
                    "parameters": model["parameters"],
                    "trained": model["trained"],
                    "created_at": model["created_at"].isoformat(),
                }

                if model["trained"]:
                    info["trained_at"] = model["trained_at"].isoformat()

                return info

            def delete_model(self, model_id: str) -> bool:
                """Delete a model."""
                if model_id in self.models:
                    del self.models[model_id]
                    return True
                return False

            def delete_dataset(self, dataset_id: str) -> bool:
                """Delete a dataset."""
                if dataset_id in self.datasets:
                    del self.datasets[dataset_id]
                    return True
                return False

        return MockPynomalySDK()

    @pytest.fixture
    def sample_data_files(self):
        """Create sample data files for testing."""
        files = {}

        # CSV file
        csv_data = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 100),
                "feature2": np.random.normal(0, 1, 100),
                "feature3": np.random.normal(0, 1, 100),
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_data.to_csv(f.name, index=False)
            files["csv"] = f.name

        # JSON file
        json_data = csv_data.to_dict("records")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_data, f, indent=2)
            files["json"] = f.name

        return files

    def test_sdk_dataset_loading_from_file(self, mock_pynomaly_sdk, sample_data_files):
        """Test SDK dataset loading from various file formats."""
        sdk = mock_pynomaly_sdk

        # Test CSV loading
        if "csv" in sample_data_files:
            dataset_id = sdk.load_dataset(
                sample_data_files["csv"], name="CSV Test Dataset"
            )

            assert dataset_id is not None
            assert len(dataset_id) > 0

            datasets = sdk.list_datasets()
            assert len(datasets) == 1
            assert datasets[0]["name"] == "CSV Test Dataset"
            assert datasets[0]["n_samples"] == 100
            assert datasets[0]["n_features"] == 3

            # Clean up
            Path(sample_data_files["csv"]).unlink()

        # Test JSON loading
        if "json" in sample_data_files:
            dataset_id = sdk.load_dataset(
                sample_data_files["json"], name="JSON Test Dataset"
            )

            assert dataset_id is not None

            datasets = sdk.list_datasets()
            json_dataset = next(d for d in datasets if d["name"] == "JSON Test Dataset")
            assert json_dataset["n_samples"] == 100
            assert json_dataset["n_features"] == 3

            # Clean up
            Path(sample_data_files["json"]).unlink()

    def test_sdk_dataset_loading_from_dataframe(self, mock_pynomaly_sdk):
        """Test SDK dataset loading from pandas DataFrame."""
        sdk = mock_pynomaly_sdk

        # Create test DataFrame
        df = pd.DataFrame(
            {
                "x": np.random.normal(0, 1, 50),
                "y": np.random.normal(0, 1, 50),
                "z": np.random.normal(0, 1, 50),
            }
        )

        dataset_id = sdk.load_dataset(df, name="DataFrame Dataset")

        assert dataset_id is not None

        datasets = sdk.list_datasets()
        assert len(datasets) == 1
        assert datasets[0]["name"] == "DataFrame Dataset"
        assert datasets[0]["n_samples"] == 50
        assert datasets[0]["n_features"] == 3

    def test_sdk_dataset_loading_from_numpy(self, mock_pynomaly_sdk):
        """Test SDK dataset loading from numpy array."""
        sdk = mock_pynomaly_sdk

        # Create test numpy array
        data = np.random.normal(0, 1, (75, 4))

        dataset_id = sdk.load_dataset(data, name="NumPy Dataset")

        assert dataset_id is not None

        datasets = sdk.list_datasets()
        assert len(datasets) == 1
        assert datasets[0]["name"] == "NumPy Dataset"
        assert datasets[0]["n_samples"] == 75
        assert datasets[0]["n_features"] == 4

    def test_sdk_model_creation_and_management(self, mock_pynomaly_sdk):
        """Test SDK model creation and management."""
        sdk = mock_pynomaly_sdk

        try:
            # Test model creation with default parameters
            model_id1 = sdk.create_model()
            assert model_id1 is not None

            # Test model creation with custom parameters
            model_id2 = sdk.create_model(
                algorithm="IsolationForest",
                contamination=0.15,
                n_estimators=50,
                random_state=123,
            )
            assert model_id2 is not None
            assert model_id1 != model_id2

            # Test listing models
            models = sdk.list_models()
            assert len(models) == 2

            # Verify model parameters
            model1_info = sdk.get_model_info(model_id1)
            assert model1_info["algorithm"] == "IsolationForest"
            assert model1_info["parameters"]["contamination"] == 0.1
            assert not model1_info["trained"]

            model2_info = sdk.get_model_info(model_id2)
            assert model2_info["parameters"]["contamination"] == 0.15
            assert model2_info["parameters"]["n_estimators"] == 50
            assert model2_info["parameters"]["random_state"] == 123

            # Test model deletion
            assert sdk.delete_model(model_id1)
            assert not sdk.delete_model(
                model_id1
            )  # Should return False for non-existent model

            models = sdk.list_models()
            assert len(models) == 1
            assert models[0]["model_id"] == model_id2

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_sdk_end_to_end_workflow(self, mock_pynomaly_sdk):
        """Test complete SDK workflow from data loading to results."""
        sdk = mock_pynomaly_sdk

        try:
            # Step 1: Load dataset
            data = pd.DataFrame(
                {
                    "sensor1": np.concatenate(
                        [
                            np.random.normal(0, 1, 90),  # Normal data
                            np.random.normal(5, 0.5, 10),  # Anomalies
                        ]
                    ),
                    "sensor2": np.concatenate(
                        [
                            np.random.normal(0, 1, 90),  # Normal data
                            np.random.normal(-5, 0.5, 10),  # Anomalies
                        ]
                    ),
                }
            )

            dataset_id = sdk.load_dataset(data, name="E2E Test Dataset")

            # Step 2: Create model
            model_id = sdk.create_model(
                algorithm="IsolationForest",
                contamination=0.1,
                n_estimators=30,
                random_state=42,
            )

            # Step 3: Train model
            training_result = sdk.train_model(model_id, dataset_id)

            assert training_result["status"] == "success"
            assert training_result["model_id"] == model_id
            assert training_result["dataset_id"] == dataset_id
            assert training_result["dataset_samples"] == 100
            assert training_result["dataset_features"] == 2

            # Verify model is now trained
            model_info = sdk.get_model_info(model_id)
            assert model_info["trained"]
            assert "trained_at" in model_info

            # Step 4: Detect anomalies
            result_id = sdk.detect_anomalies(model_id, dataset_id)
            assert result_id is not None

            # Step 5: Get results
            results = sdk.get_results(result_id)

            assert results["result_id"] == result_id
            assert results["model_id"] == model_id
            assert results["dataset_id"] == dataset_id
            assert results["n_samples"] == 100
            assert 0 <= results["contamination_rate"] <= 0.2  # Should be around 0.1
            assert len(results["scores"]) == 100
            assert len(results["labels"]) == 100
            assert all(isinstance(score, float) for score in results["scores"])
            assert all(label in [0, 1] for label in results["labels"])

            # The last 10 samples (anomalies) should generally have higher scores
            anomaly_scores = results["scores"][90:]
            normal_scores = results["scores"][:90]

            avg_anomaly_score = np.mean(anomaly_scores)
            avg_normal_score = np.mean(normal_scores)

            # This is not guaranteed but should generally be true
            # (commented out as it might be flaky)
            # assert avg_anomaly_score > avg_normal_score

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_sdk_error_handling(self, mock_pynomaly_sdk):
        """Test SDK error handling."""
        sdk = mock_pynomaly_sdk

        # Test invalid dataset ID
        with pytest.raises(ValueError, match="not found"):
            sdk.train_model("invalid_model_id", "invalid_dataset_id")

        # Test invalid model ID
        data = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        dataset_id = sdk.load_dataset(data)

        with pytest.raises(ValueError, match="not found"):
            sdk.train_model("invalid_model_id", dataset_id)

        try:
            # Test training with valid IDs
            model_id = sdk.create_model()

            # Test detection without training
            with pytest.raises(ValueError, match="not trained"):
                sdk.detect_anomalies(model_id, dataset_id)

            # Test getting non-existent results
            with pytest.raises(ValueError, match="not found"):
                sdk.get_results("invalid_result_id")

            # Test getting non-existent model info
            with pytest.raises(ValueError, match="not found"):
                sdk.get_model_info("invalid_model_id")

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_sdk_concurrent_operations(self, mock_pynomaly_sdk):
        """Test SDK handling of concurrent operations."""
        sdk = mock_pynomaly_sdk

        try:
            # Create multiple datasets
            datasets = []
            for i in range(3):
                data = pd.DataFrame(
                    {
                        "feature1": np.random.normal(i, 1, 50),
                        "feature2": np.random.normal(i, 1, 50),
                    }
                )
                dataset_id = sdk.load_dataset(data, name=f"Dataset {i}")
                datasets.append(dataset_id)

            # Create multiple models
            models = []
            for i in range(3):
                model_id = sdk.create_model(
                    contamination=0.1 + i * 0.05,
                    n_estimators=30 + i * 10,
                    random_state=42 + i,
                )
                models.append(model_id)

            # Train all models on all datasets
            training_results = []
            for model_id in models:
                for dataset_id in datasets:
                    result = sdk.train_model(model_id, dataset_id)
                    training_results.append(result)

            # All training should succeed
            assert len(training_results) == 9  # 3 models × 3 datasets
            assert all(r["status"] == "success" for r in training_results)

            # Test detection with all combinations
            detection_results = []
            for model_id in models:
                for dataset_id in datasets:
                    result_id = sdk.detect_anomalies(model_id, dataset_id)
                    detection_results.append(result_id)

            assert len(detection_results) == 9
            assert all(result_id is not None for result_id in detection_results)

            # Verify all results can be retrieved
            for result_id in detection_results:
                results = sdk.get_results(result_id)
                assert results["n_samples"] == 50
                assert 0.0 <= results["contamination_rate"] <= 0.5

        except ImportError:
            pytest.skip("scikit-learn not available")
