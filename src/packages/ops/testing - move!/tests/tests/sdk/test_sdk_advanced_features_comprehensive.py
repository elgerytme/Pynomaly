"""Comprehensive SDK advanced features tests.

This module contains comprehensive tests for advanced SDK features including
ensemble methods, model persistence, streaming detection, configuration
management, and performance optimization.
"""

import json
import pickle
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from monorepo.domain.entities import Dataset
from monorepo.infrastructure.adapters.sklearn_adapter import SklearnAdapter


class TestSDKEnsembleMethods:
    """Test SDK ensemble detection methods."""

    @pytest.fixture
    def mock_ensemble_sdk(self):
        """Create mock SDK with ensemble capabilities."""

        class MockEnsembleSDK:
            def __init__(self):
                self.models = {}
                self.datasets = {}
                self.ensembles = {}
                self.results = {}

            def create_ensemble(self, ensemble_config: dict[str, Any]) -> str:
                """Create ensemble detector."""
                ensemble_id = str(uuid.uuid4())

                # Create individual models for ensemble
                models = {}
                for model_name, config in ensemble_config.get("models", {}).items():
                    try:
                        adapter = SklearnAdapter(
                            algorithm_name=config["algorithm"],
                            parameters=config["parameters"],
                        )
                        models[model_name] = {
                            "adapter": adapter,
                            "weight": config.get("weight", 1.0),
                            "config": config,
                        }
                    except Exception:
                        continue

                if not models:
                    raise ValueError("No valid models could be created for ensemble")

                self.ensembles[ensemble_id] = {
                    "models": models,
                    "aggregation_method": ensemble_config.get("aggregation", "voting"),
                    "voting_threshold": ensemble_config.get("voting_threshold", 0.5),
                    "trained": False,
                    "created_at": datetime.utcnow(),
                }

                return ensemble_id

            def train_ensemble(
                self, ensemble_id: str, dataset_id: str
            ) -> dict[str, Any]:
                """Train ensemble on dataset."""
                if ensemble_id not in self.ensembles:
                    raise ValueError(f"Ensemble {ensemble_id} not found")

                if dataset_id not in self.datasets:
                    raise ValueError(f"Dataset {dataset_id} not found")

                ensemble = self.ensembles[ensemble_id]
                dataset = self.datasets[dataset_id]

                training_results = {}

                for model_name, model_data in ensemble["models"].items():
                    try:
                        adapter = model_data["adapter"]
                        adapter.fit(dataset)
                        training_results[model_name] = {"success": True}
                    except Exception as e:
                        training_results[model_name] = {
                            "success": False,
                            "error": str(e),
                        }

                successful_models = sum(
                    1 for r in training_results.values() if r["success"]
                )

                if successful_models == 0:
                    return {
                        "ensemble_id": ensemble_id,
                        "status": "failed",
                        "error": "No models trained successfully",
                        "model_results": training_results,
                    }

                ensemble["trained"] = True
                ensemble["trained_at"] = datetime.utcnow()

                return {
                    "ensemble_id": ensemble_id,
                    "status": "success",
                    "successful_models": successful_models,
                    "total_models": len(ensemble["models"]),
                    "model_results": training_results,
                    "trained_at": ensemble["trained_at"].isoformat(),
                }

            def detect_with_ensemble(self, ensemble_id: str, dataset_id: str) -> str:
                """Detect anomalies using ensemble."""
                if ensemble_id not in self.ensembles:
                    raise ValueError(f"Ensemble {ensemble_id} not found")

                if dataset_id not in self.datasets:
                    raise ValueError(f"Dataset {dataset_id} not found")

                ensemble = self.ensembles[ensemble_id]
                dataset = self.datasets[dataset_id]

                if not ensemble["trained"]:
                    raise ValueError(f"Ensemble {ensemble_id} is not trained")

                # Get predictions from all models
                model_predictions = {}
                model_scores = {}

                for model_name, model_data in ensemble["models"].items():
                    try:
                        adapter = model_data["adapter"]
                        if adapter.is_fitted:
                            scores = adapter.score(dataset)
                            result = adapter.detect(dataset)

                            model_predictions[model_name] = result.labels
                            model_scores[model_name] = [score.value for score in scores]
                    except Exception:
                        continue

                if not model_predictions:
                    raise RuntimeError("No model predictions available")

                # Aggregate predictions
                aggregation_method = ensemble["aggregation_method"]

                if aggregation_method == "voting":
                    ensemble_labels = self._aggregate_voting(
                        model_predictions, ensemble["voting_threshold"]
                    )
                    ensemble_scores = self._aggregate_scores_average(model_scores)
                elif aggregation_method == "weighted_average":
                    weights = {
                        name: data["weight"]
                        for name, data in ensemble["models"].items()
                    }
                    ensemble_scores = self._aggregate_scores_weighted(
                        model_scores, weights
                    )
                    ensemble_labels = (np.array(ensemble_scores) > 0.5).astype(int)
                else:
                    ensemble_scores = self._aggregate_scores_average(model_scores)
                    ensemble_labels = (np.array(ensemble_scores) > 0.5).astype(int)

                # Store results
                result_id = str(uuid.uuid4())
                self.results[result_id] = {
                    "ensemble_id": ensemble_id,
                    "dataset_id": dataset_id,
                    "scores": ensemble_scores,
                    "labels": ensemble_labels.tolist(),
                    "model_predictions": model_predictions,
                    "model_scores": model_scores,
                    "aggregation_method": aggregation_method,
                    "created_at": datetime.utcnow(),
                }

                return result_id

            def _aggregate_voting(
                self, predictions: dict[str, np.ndarray], threshold: float
            ) -> np.ndarray:
                """Aggregate predictions using majority voting."""
                prediction_matrix = np.array(list(predictions.values()))
                vote_counts = np.sum(prediction_matrix, axis=0)
                total_models = len(predictions)

                return (vote_counts / total_models > threshold).astype(int)

            def _aggregate_scores_average(
                self, scores: dict[str, list[float]]
            ) -> list[float]:
                """Aggregate scores using simple average."""
                if not scores:
                    return []

                score_matrix = np.array(list(scores.values()))
                return np.mean(score_matrix, axis=0).tolist()

            def _aggregate_scores_weighted(
                self, scores: dict[str, list[float]], weights: dict[str, float]
            ) -> list[float]:
                """Aggregate scores using weighted average."""
                if not scores:
                    return []

                weighted_scores = []
                total_weight = 0

                for model_name, model_scores in scores.items():
                    weight = weights.get(model_name, 1.0)
                    if len(weighted_scores) == 0:
                        weighted_scores = np.array(model_scores) * weight
                    else:
                        weighted_scores += np.array(model_scores) * weight
                    total_weight += weight

                return (weighted_scores / total_weight).tolist()

            def load_dataset(
                self, data: pd.DataFrame | np.ndarray, name: str = None
            ) -> str:
                """Load dataset for testing."""
                if isinstance(data, np.ndarray):
                    data = pd.DataFrame(data)

                dataset_id = str(uuid.uuid4())
                dataset_name = name or f"dataset_{dataset_id[:8]}"
                self.datasets[dataset_id] = Dataset(name=dataset_name, data=data)

                return dataset_id

            def get_ensemble_results(self, result_id: str) -> dict[str, Any]:
                """Get ensemble detection results."""
                if result_id not in self.results:
                    raise ValueError(f"Results {result_id} not found")

                result_data = self.results[result_id]

                return {
                    "result_id": result_id,
                    "ensemble_id": result_data["ensemble_id"],
                    "dataset_id": result_data["dataset_id"],
                    "n_samples": len(result_data["labels"]),
                    "n_anomalies": int(np.sum(result_data["labels"])),
                    "contamination_rate": float(np.mean(result_data["labels"])),
                    "ensemble_scores": result_data["scores"],
                    "ensemble_labels": result_data["labels"],
                    "aggregation_method": result_data["aggregation_method"],
                    "model_count": len(result_data["model_predictions"]),
                    "created_at": result_data["created_at"].isoformat(),
                }

        return MockEnsembleSDK()

    def test_ensemble_creation_and_training(self, mock_ensemble_sdk):
        """Test ensemble creation and training."""
        try:
            sdk = mock_ensemble_sdk

            # Create test dataset
            data = pd.DataFrame(
                {
                    "feature1": np.random.normal(0, 1, 200),
                    "feature2": np.random.normal(0, 1, 200),
                    "feature3": np.random.normal(0, 1, 200),
                }
            )
            dataset_id = sdk.load_dataset(data, "Ensemble Test Dataset")

            # Define ensemble configuration
            ensemble_config = {
                "models": {
                    "isolation_forest_1": {
                        "algorithm": "IsolationForest",
                        "parameters": {
                            "contamination": 0.1,
                            "n_estimators": 30,
                            "random_state": 42,
                        },
                        "weight": 1.0,
                    },
                    "isolation_forest_2": {
                        "algorithm": "IsolationForest",
                        "parameters": {
                            "contamination": 0.15,
                            "n_estimators": 40,
                            "random_state": 123,
                        },
                        "weight": 1.5,
                    },
                    "local_outlier_factor": {
                        "algorithm": "LocalOutlierFactor",
                        "parameters": {
                            "contamination": 0.1,
                            "n_neighbors": 20,
                            "novelty": True,
                        },
                        "weight": 0.8,
                    },
                },
                "aggregation": "voting",
                "voting_threshold": 0.5,
            }

            # Create ensemble
            ensemble_id = sdk.create_ensemble(ensemble_config)
            assert ensemble_id is not None

            # Train ensemble
            training_result = sdk.train_ensemble(ensemble_id, dataset_id)

            assert training_result["status"] == "success"
            assert training_result["ensemble_id"] == ensemble_id
            assert training_result["successful_models"] > 0
            assert (
                training_result["successful_models"] <= training_result["total_models"]
            )

            # Test detection
            result_id = sdk.detect_with_ensemble(ensemble_id, dataset_id)
            assert result_id is not None

            # Get results
            results = sdk.get_ensemble_results(result_id)

            assert results["ensemble_id"] == ensemble_id
            assert results["dataset_id"] == dataset_id
            assert results["n_samples"] == 200
            assert 0.0 <= results["contamination_rate"] <= 0.3
            assert len(results["ensemble_scores"]) == 200
            assert len(results["ensemble_labels"]) == 200
            assert results["aggregation_method"] == "voting"
            assert results["model_count"] > 0

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_ensemble_aggregation_methods(self, mock_ensemble_sdk):
        """Test different ensemble aggregation methods."""
        try:
            sdk = mock_ensemble_sdk

            # Create test dataset
            data = pd.DataFrame(
                {"x": np.random.normal(0, 1, 100), "y": np.random.normal(0, 1, 100)}
            )
            dataset_id = sdk.load_dataset(data, "Aggregation Test")

            aggregation_methods = ["voting", "weighted_average"]

            for method in aggregation_methods:
                # Create ensemble with different aggregation method
                ensemble_config = {
                    "models": {
                        "model_1": {
                            "algorithm": "IsolationForest",
                            "parameters": {
                                "contamination": 0.1,
                                "n_estimators": 20,
                                "random_state": 42,
                            },
                            "weight": 1.0,
                        },
                        "model_2": {
                            "algorithm": "IsolationForest",
                            "parameters": {
                                "contamination": 0.15,
                                "n_estimators": 25,
                                "random_state": 123,
                            },
                            "weight": 2.0,
                        },
                    },
                    "aggregation": method,
                    "voting_threshold": 0.3,
                }

                ensemble_id = sdk.create_ensemble(ensemble_config)
                training_result = sdk.train_ensemble(ensemble_id, dataset_id)

                if training_result["status"] == "success":
                    result_id = sdk.detect_with_ensemble(ensemble_id, dataset_id)
                    results = sdk.get_ensemble_results(result_id)

                    assert results["aggregation_method"] == method
                    assert len(results["ensemble_scores"]) == 100
                    assert len(results["ensemble_labels"]) == 100

                    # Verify scores are in valid range
                    for score in results["ensemble_scores"]:
                        assert 0.0 <= score <= 1.0

                    # Verify labels are binary
                    for label in results["ensemble_labels"]:
                        assert label in [0, 1]

        except ImportError:
            pytest.skip("scikit-learn not available")


class TestSDKModelPersistence:
    """Test SDK model persistence and serialization."""

    @pytest.fixture
    def mock_persistence_sdk(self):
        """Create mock SDK with persistence capabilities."""

        class MockPersistenceSDK:
            def __init__(self):
                self.models = {}
                self.datasets = {}

            def create_model(self, algorithm: str, **parameters) -> str:
                """Create a model."""
                try:
                    adapter = SklearnAdapter(
                        algorithm_name=algorithm, parameters=parameters
                    )

                    model_id = str(uuid.uuid4())
                    self.models[model_id] = {
                        "adapter": adapter,
                        "algorithm": algorithm,
                        "parameters": parameters,
                        "trained": False,
                        "created_at": datetime.utcnow(),
                    }

                    return model_id
                except Exception as e:
                    raise ValueError(f"Failed to create model: {str(e)}")

            def train_model(self, model_id: str, dataset_id: str) -> bool:
                """Train a model."""
                if model_id not in self.models or dataset_id not in self.datasets:
                    return False

                try:
                    model = self.models[model_id]
                    dataset = self.datasets[dataset_id]

                    adapter = model["adapter"]
                    adapter.fit(dataset)

                    model["trained"] = True
                    model["trained_at"] = datetime.utcnow()

                    return True
                except Exception:
                    return False

            def save_model(self, model_id: str, file_path: str) -> bool:
                """Save model to file."""
                if model_id not in self.models:
                    return False

                try:
                    model_data = self.models[model_id]

                    # Create serializable model data
                    save_data = {
                        "model_id": model_id,
                        "algorithm": model_data["algorithm"],
                        "parameters": model_data["parameters"],
                        "trained": model_data["trained"],
                        "created_at": model_data["created_at"].isoformat(),
                        "adapter": model_data[
                            "adapter"
                        ],  # This includes the trained model
                    }

                    if model_data["trained"]:
                        save_data["trained_at"] = model_data["trained_at"].isoformat()

                    # Save using pickle
                    with open(file_path, "wb") as f:
                        pickle.dump(save_data, f)

                    return True
                except Exception:
                    return False

            def load_model(self, file_path: str) -> str:
                """Load model from file."""
                try:
                    with open(file_path, "rb") as f:
                        save_data = pickle.load(f)

                    model_id = save_data["model_id"]

                    # Reconstruct model data
                    self.models[model_id] = {
                        "adapter": save_data["adapter"],
                        "algorithm": save_data["algorithm"],
                        "parameters": save_data["parameters"],
                        "trained": save_data["trained"],
                        "created_at": datetime.fromisoformat(save_data["created_at"]),
                    }

                    if save_data["trained"]:
                        self.models[model_id]["trained_at"] = datetime.fromisoformat(
                            save_data["trained_at"]
                        )

                    return model_id
                except Exception:
                    return None

            def export_model_config(self, model_id: str, config_path: str) -> bool:
                """Export model configuration to JSON."""
                if model_id not in self.models:
                    return False

                try:
                    model = self.models[model_id]

                    config = {
                        "model_id": model_id,
                        "algorithm": model["algorithm"],
                        "parameters": model["parameters"],
                        "trained": model["trained"],
                        "created_at": model["created_at"].isoformat(),
                    }

                    if model["trained"]:
                        config["trained_at"] = model["trained_at"].isoformat()

                    with open(config_path, "w") as f:
                        json.dump(config, f, indent=2)

                    return True
                except Exception:
                    return False

            def load_dataset(self, data: pd.DataFrame, name: str = None) -> str:
                """Load dataset for testing."""
                dataset_id = str(uuid.uuid4())
                dataset_name = name or f"dataset_{dataset_id[:8]}"
                self.datasets[dataset_id] = Dataset(name=dataset_name, data=data)
                return dataset_id

            def get_model_info(self, model_id: str) -> dict[str, Any] | None:
                """Get model information."""
                if model_id not in self.models:
                    return None

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

        return MockPersistenceSDK()

    def test_model_save_and_load(self, mock_persistence_sdk):
        """Test model saving and loading."""
        try:
            sdk = mock_persistence_sdk

            # Create and train model
            model_id = sdk.create_model(
                algorithm="IsolationForest",
                contamination=0.1,
                n_estimators=50,
                random_state=42,
            )

            # Create dataset
            data = pd.DataFrame(
                {
                    "feature1": np.random.normal(0, 1, 100),
                    "feature2": np.random.normal(0, 1, 100),
                }
            )
            dataset_id = sdk.load_dataset(data, "Persistence Test")

            # Train model
            training_success = sdk.train_model(model_id, dataset_id)
            assert training_success

            # Get model info before saving
            original_info = sdk.get_model_info(model_id)
            assert original_info["trained"]

            # Save model
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                model_path = f.name

            save_success = sdk.save_model(model_id, model_path)
            assert save_success

            # Remove original model from SDK
            del sdk.models[model_id]

            # Load model back
            loaded_model_id = sdk.load_model(model_path)
            assert loaded_model_id is not None
            assert loaded_model_id == model_id

            # Verify loaded model info
            loaded_info = sdk.get_model_info(loaded_model_id)
            assert loaded_info is not None
            assert loaded_info["algorithm"] == original_info["algorithm"]
            assert loaded_info["parameters"] == original_info["parameters"]
            assert loaded_info["trained"] == original_info["trained"]
            assert loaded_info["created_at"] == original_info["created_at"]

            # Clean up
            Path(model_path).unlink()

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_model_config_export(self, mock_persistence_sdk):
        """Test model configuration export."""
        try:
            sdk = mock_persistence_sdk

            # Create model
            model_id = sdk.create_model(
                algorithm="LocalOutlierFactor",
                contamination=0.15,
                n_neighbors=25,
                novelty=True,
            )

            # Export configuration
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                config_path = f.name

            export_success = sdk.export_model_config(model_id, config_path)
            assert export_success

            # Load and verify configuration
            with open(config_path) as f:
                config = json.load(f)

            assert config["model_id"] == model_id
            assert config["algorithm"] == "LocalOutlierFactor"
            assert config["parameters"]["contamination"] == 0.15
            assert config["parameters"]["n_neighbors"] == 25
            assert config["parameters"]["novelty"] is True
            assert config["trained"] is False
            assert "created_at" in config

            # Clean up
            Path(config_path).unlink()

        except ImportError:
            pytest.skip("scikit-learn not available")


class TestSDKPerformanceOptimization:
    """Test SDK performance optimization features."""

    @pytest.fixture
    def mock_performance_sdk(self):
        """Create mock SDK with performance features."""

        class MockPerformanceSDK:
            def __init__(self):
                self.models = {}
                self.datasets = {}
                self.cache = {}
                self.performance_metrics = {}

            def create_model(self, algorithm: str, **parameters) -> str:
                """Create optimized model."""
                try:
                    adapter = SklearnAdapter(
                        algorithm_name=algorithm, parameters=parameters
                    )

                    model_id = str(uuid.uuid4())
                    self.models[model_id] = {
                        "adapter": adapter,
                        "algorithm": algorithm,
                        "parameters": parameters,
                        "trained": False,
                        "performance_profile": {
                            "training_time": None,
                            "prediction_time": None,
                            "memory_usage": None,
                        },
                    }

                    return model_id
                except Exception as e:
                    raise ValueError(f"Failed to create model: {str(e)}")

            def train_model_with_profiling(
                self, model_id: str, dataset_id: str
            ) -> dict[str, Any]:
                """Train model with performance profiling."""
                if model_id not in self.models or dataset_id not in self.datasets:
                    raise ValueError("Model or dataset not found")

                model = self.models[model_id]
                dataset = self.datasets[dataset_id]

                # Profile training
                start_time = time.time()
                start_memory = self._get_memory_usage()

                try:
                    adapter = model["adapter"]
                    adapter.fit(dataset)

                    end_time = time.time()
                    end_memory = self._get_memory_usage()

                    training_time = end_time - start_time
                    memory_delta = end_memory - start_memory

                    model["trained"] = True
                    model["performance_profile"]["training_time"] = training_time
                    model["performance_profile"]["memory_usage"] = memory_delta

                    return {
                        "model_id": model_id,
                        "training_time": training_time,
                        "memory_usage_mb": memory_delta / (1024 * 1024),
                        "dataset_samples": dataset.n_samples,
                        "dataset_features": dataset.n_features,
                        "samples_per_second": (
                            dataset.n_samples / training_time
                            if training_time > 0
                            else 0
                        ),
                    }

                except Exception as e:
                    return {
                        "model_id": model_id,
                        "error": str(e),
                        "training_time": None,
                    }

            def predict_with_profiling(
                self, model_id: str, dataset_id: str
            ) -> dict[str, Any]:
                """Make predictions with performance profiling."""
                if model_id not in self.models or dataset_id not in self.datasets:
                    raise ValueError("Model or dataset not found")

                model = self.models[model_id]
                dataset = self.datasets[dataset_id]

                if not model["trained"]:
                    raise ValueError("Model is not trained")

                # Profile prediction
                start_time = time.time()

                try:
                    adapter = model["adapter"]
                    scores = adapter.score(dataset)
                    result = adapter.detect(dataset)

                    end_time = time.time()
                    prediction_time = end_time - start_time

                    model["performance_profile"]["prediction_time"] = prediction_time

                    return {
                        "model_id": model_id,
                        "prediction_time": prediction_time,
                        "samples_predicted": len(scores),
                        "predictions_per_second": (
                            len(scores) / prediction_time if prediction_time > 0 else 0
                        ),
                        "contamination_rate": float(np.mean(result.labels)),
                        "avg_score": float(np.mean([score.value for score in scores])),
                    }

                except Exception as e:
                    return {
                        "model_id": model_id,
                        "error": str(e),
                        "prediction_time": None,
                    }

            def batch_predict(
                self, model_id: str, datasets: list[str], batch_size: int = None
            ) -> list[dict[str, Any]]:
                """Perform batch predictions for performance."""
                if model_id not in self.models:
                    raise ValueError("Model not found")

                model = self.models[model_id]
                if not model["trained"]:
                    raise ValueError("Model is not trained")

                results = []

                # Process datasets in batches if batch_size specified
                if batch_size:
                    for i in range(0, len(datasets), batch_size):
                        batch = datasets[i : i + batch_size]
                        batch_results = self._process_batch(model_id, batch)
                        results.extend(batch_results)
                else:
                    results = self._process_batch(model_id, datasets)

                return results

            def _process_batch(
                self, model_id: str, dataset_ids: list[str]
            ) -> list[dict[str, Any]]:
                """Process a batch of datasets."""
                model = self.models[model_id]
                adapter = model["adapter"]

                batch_results = []

                for dataset_id in dataset_ids:
                    if dataset_id not in self.datasets:
                        batch_results.append(
                            {"dataset_id": dataset_id, "error": "Dataset not found"}
                        )
                        continue

                    dataset = self.datasets[dataset_id]

                    try:
                        start_time = time.time()
                        scores = adapter.score(dataset)
                        result = adapter.detect(dataset)
                        end_time = time.time()

                        batch_results.append(
                            {
                                "dataset_id": dataset_id,
                                "prediction_time": end_time - start_time,
                                "n_samples": len(scores),
                                "contamination_rate": float(np.mean(result.labels)),
                                "success": True,
                            }
                        )

                    except Exception as e:
                        batch_results.append(
                            {
                                "dataset_id": dataset_id,
                                "error": str(e),
                                "success": False,
                            }
                        )

                return batch_results

            def get_performance_metrics(self, model_id: str) -> dict[str, Any]:
                """Get comprehensive performance metrics."""
                if model_id not in self.models:
                    raise ValueError("Model not found")

                model = self.models[model_id]
                profile = model["performance_profile"]

                metrics = {
                    "model_id": model_id,
                    "algorithm": model["algorithm"],
                    "trained": model["trained"],
                    "training_time": profile["training_time"],
                    "prediction_time": profile["prediction_time"],
                    "memory_usage_bytes": profile["memory_usage"],
                }

                # Add derived metrics
                if profile["training_time"] and profile["prediction_time"]:
                    metrics["speed_ratio"] = (
                        profile["training_time"] / profile["prediction_time"]
                    )

                if profile["memory_usage"]:
                    metrics["memory_usage_mb"] = profile["memory_usage"] / (1024 * 1024)

                return metrics

            def _get_memory_usage(self) -> int:
                """Get current memory usage (mock implementation)."""
                try:
                    import psutil

                    return psutil.Process().memory_info().rss
                except ImportError:
                    return 0  # Mock value

            def load_dataset(self, data: pd.DataFrame, name: str = None) -> str:
                """Load dataset."""
                dataset_id = str(uuid.uuid4())
                dataset_name = name or f"dataset_{dataset_id[:8]}"
                self.datasets[dataset_id] = Dataset(name=dataset_name, data=data)
                return dataset_id

        return MockPerformanceSDK()

    def test_model_training_profiling(self, mock_performance_sdk):
        """Test model training with performance profiling."""
        try:
            sdk = mock_performance_sdk

            # Create model
            model_id = sdk.create_model(
                algorithm="IsolationForest",
                contamination=0.1,
                n_estimators=50,
                random_state=42,
            )

            # Create dataset
            data = pd.DataFrame(
                {
                    "feature1": np.random.normal(0, 1, 1000),
                    "feature2": np.random.normal(0, 1, 1000),
                    "feature3": np.random.normal(0, 1, 1000),
                }
            )
            dataset_id = sdk.load_dataset(data, "Performance Test")

            # Train with profiling
            training_result = sdk.train_model_with_profiling(model_id, dataset_id)

            assert training_result["model_id"] == model_id
            assert training_result["training_time"] > 0
            assert training_result["dataset_samples"] == 1000
            assert training_result["dataset_features"] == 3
            assert training_result["samples_per_second"] > 0

            # Verify performance metrics
            metrics = sdk.get_performance_metrics(model_id)
            assert metrics["model_id"] == model_id
            assert metrics["algorithm"] == "IsolationForest"
            assert metrics["trained"] is True
            assert metrics["training_time"] > 0

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_prediction_profiling(self, mock_performance_sdk):
        """Test prediction performance profiling."""
        try:
            sdk = mock_performance_sdk

            # Create and train model
            model_id = sdk.create_model(
                algorithm="IsolationForest",
                contamination=0.1,
                n_estimators=30,
                random_state=42,
            )

            # Training dataset
            train_data = pd.DataFrame(
                {"x": np.random.normal(0, 1, 500), "y": np.random.normal(0, 1, 500)}
            )
            train_dataset_id = sdk.load_dataset(train_data, "Training Data")

            # Train model
            training_result = sdk.train_model_with_profiling(model_id, train_dataset_id)
            assert training_result["model_id"] == model_id

            # Prediction dataset
            pred_data = pd.DataFrame(
                {"x": np.random.normal(0, 1, 200), "y": np.random.normal(0, 1, 200)}
            )
            pred_dataset_id = sdk.load_dataset(pred_data, "Prediction Data")

            # Predict with profiling
            prediction_result = sdk.predict_with_profiling(model_id, pred_dataset_id)

            assert prediction_result["model_id"] == model_id
            assert prediction_result["prediction_time"] > 0
            assert prediction_result["samples_predicted"] == 200
            assert prediction_result["predictions_per_second"] > 0
            assert 0.0 <= prediction_result["contamination_rate"] <= 0.5
            assert 0.0 <= prediction_result["avg_score"] <= 1.0

            # Verify complete performance metrics
            metrics = sdk.get_performance_metrics(model_id)
            assert metrics["training_time"] > 0
            assert metrics["prediction_time"] > 0
            assert "speed_ratio" in metrics

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_batch_prediction_performance(self, mock_performance_sdk):
        """Test batch prediction performance."""
        try:
            sdk = mock_performance_sdk

            # Create and train model
            model_id = sdk.create_model(
                algorithm="IsolationForest",
                contamination=0.1,
                n_estimators=20,
                random_state=42,
            )

            # Training dataset
            train_data = pd.DataFrame(
                {
                    "feature1": np.random.normal(0, 1, 300),
                    "feature2": np.random.normal(0, 1, 300),
                }
            )
            train_dataset_id = sdk.load_dataset(train_data, "Batch Training")

            # Train model
            training_result = sdk.train_model_with_profiling(model_id, train_dataset_id)
            assert training_result["model_id"] == model_id

            # Create multiple prediction datasets
            prediction_datasets = []
            for i in range(5):
                data = pd.DataFrame(
                    {
                        "feature1": np.random.normal(i * 0.5, 1, 100),
                        "feature2": np.random.normal(i * 0.5, 1, 100),
                    }
                )
                dataset_id = sdk.load_dataset(data, f"Batch Dataset {i}")
                prediction_datasets.append(dataset_id)

            # Test batch prediction
            batch_results = sdk.batch_predict(model_id, prediction_datasets)

            assert len(batch_results) == 5

            for result in batch_results:
                assert result["success"] is True
                assert result["n_samples"] == 100
                assert result["prediction_time"] > 0
                assert 0.0 <= result["contamination_rate"] <= 0.5

            # Test batch prediction with batch size
            batch_results_sized = sdk.batch_predict(
                model_id, prediction_datasets, batch_size=2
            )

            assert len(batch_results_sized) == 5

            # Results should be similar to non-batched version
            for i, result in enumerate(batch_results_sized):
                assert result["success"] is True
                assert result["n_samples"] == 100

        except ImportError:
            pytest.skip("scikit-learn not available")
