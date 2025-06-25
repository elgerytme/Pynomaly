"""Comprehensive SDK error handling and edge cases tests.

This module contains comprehensive tests for SDK error handling,
edge cases, graceful degradation, and recovery mechanisms.
"""

import threading
import time
import uuid
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pynomaly.domain.entities import Dataset
from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter


class TestSDKErrorHandling:
    """Test SDK error handling and recovery mechanisms."""

    @pytest.fixture
    def mock_robust_sdk(self):
        """Create mock SDK with robust error handling."""

        class MockRobustSDK:
            def __init__(self):
                self.models = {}
                self.datasets = {}
                self.error_log = []
                self.recovery_attempts = {}
                self.circuit_breakers = {}
                self.retry_policies = {
                    "max_retries": 3,
                    "retry_delay": 0.1,
                    "exponential_backoff": True,
                }

            def create_model_with_fallback(
                self,
                primary_config: dict[str, Any],
                fallback_configs: list[dict[str, Any]] = None,
            ) -> dict[str, Any]:
                """Create model with fallback configurations."""
                model_id = str(uuid.uuid4())

                # Try primary configuration
                creation_result = self._try_create_model(
                    model_id, primary_config, "primary"
                )

                if creation_result["success"]:
                    return {
                        "model_id": model_id,
                        "status": "success",
                        "config_used": "primary",
                        "config": primary_config,
                    }

                # Try fallback configurations
                if fallback_configs:
                    for i, fallback_config in enumerate(fallback_configs):
                        fallback_result = self._try_create_model(
                            model_id, fallback_config, f"fallback_{i}"
                        )

                        if fallback_result["success"]:
                            return {
                                "model_id": model_id,
                                "status": "success_with_fallback",
                                "config_used": f"fallback_{i}",
                                "config": fallback_config,
                                "primary_error": creation_result["error"],
                            }

                # All configurations failed
                return {
                    "model_id": None,
                    "status": "failed",
                    "primary_error": creation_result["error"],
                    "fallback_errors": [
                        self._try_create_model(model_id, cfg, f"fallback_{i}")["error"]
                        for i, cfg in enumerate(fallback_configs or [])
                    ],
                }

            def _try_create_model(
                self, model_id: str, config: dict[str, Any], config_name: str
            ) -> dict[str, Any]:
                """Try to create model with given configuration."""
                try:
                    algorithm = config.get("algorithm", "IsolationForest")
                    parameters = {k: v for k, v in config.items() if k != "algorithm"}

                    adapter = SklearnAdapter(
                        algorithm_name=algorithm, parameters=parameters
                    )

                    self.models[model_id] = {
                        "adapter": adapter,
                        "config": config,
                        "config_name": config_name,
                        "created_at": time.time(),
                        "trained": False,
                        "error_count": 0,
                    }

                    return {"success": True, "error": None}

                except Exception as e:
                    error_msg = (
                        f"Failed to create model with {config_name} config: {str(e)}"
                    )
                    self._log_error(
                        error_msg, {"config": config, "config_name": config_name}
                    )
                    return {"success": False, "error": error_msg}

            def train_with_retry(
                self, model_id: str, dataset_id: str
            ) -> dict[str, Any]:
                """Train model with retry mechanism."""
                if model_id not in self.models:
                    return {"success": False, "error": "Model not found", "retries": 0}

                if dataset_id not in self.datasets:
                    return {
                        "success": False,
                        "error": "Dataset not found",
                        "retries": 0,
                    }

                model = self.models[model_id]
                dataset = self.datasets[dataset_id]

                max_retries = self.retry_policies["max_retries"]
                retry_delay = self.retry_policies["retry_delay"]

                for attempt in range(max_retries + 1):
                    try:
                        adapter = model["adapter"]
                        adapter.fit(dataset)

                        model["trained"] = True
                        model["trained_at"] = time.time()
                        model["error_count"] = 0

                        return {
                            "success": True,
                            "model_id": model_id,
                            "dataset_id": dataset_id,
                            "retries": attempt,
                            "training_time": time.time() - model["created_at"],
                        }

                    except Exception as e:
                        error_msg = f"Training attempt {attempt + 1} failed: {str(e)}"
                        self._log_error(
                            error_msg,
                            {
                                "model_id": model_id,
                                "dataset_id": dataset_id,
                                "attempt": attempt,
                            },
                        )

                        model["error_count"] += 1

                        if attempt < max_retries:
                            # Calculate retry delay with exponential backoff
                            if self.retry_policies["exponential_backoff"]:
                                delay = retry_delay * (2**attempt)
                            else:
                                delay = retry_delay

                            time.sleep(delay)
                        else:
                            return {
                                "success": False,
                                "error": error_msg,
                                "retries": attempt,
                                "total_errors": model["error_count"],
                            }

                return {
                    "success": False,
                    "error": "Maximum retries exceeded",
                    "retries": max_retries,
                }

            def predict_with_circuit_breaker(
                self, model_id: str, dataset_id: str
            ) -> dict[str, Any]:
                """Make predictions with circuit breaker pattern."""
                circuit_key = f"{model_id}_{dataset_id}"

                # Check circuit breaker state
                circuit_state = self._get_circuit_breaker_state(circuit_key)

                if circuit_state == "open":
                    return {
                        "success": False,
                        "error": "Circuit breaker is open",
                        "circuit_state": "open",
                    }

                elif circuit_state == "half_open":
                    # Try one request to test if service is recovered
                    result = self._try_prediction(model_id, dataset_id)

                    if result["success"]:
                        self._close_circuit_breaker(circuit_key)
                        result["circuit_state"] = "closed"
                    else:
                        self._open_circuit_breaker(circuit_key)
                        result["circuit_state"] = "open"

                    return result

                else:  # circuit_state == "closed"
                    result = self._try_prediction(model_id, dataset_id)

                    if not result["success"]:
                        self._record_circuit_failure(circuit_key)
                    else:
                        self._record_circuit_success(circuit_key)

                    result["circuit_state"] = self._get_circuit_breaker_state(
                        circuit_key
                    )
                    return result

            def _try_prediction(self, model_id: str, dataset_id: str) -> dict[str, Any]:
                """Attempt prediction with error handling."""
                try:
                    if model_id not in self.models:
                        raise ValueError("Model not found")

                    if dataset_id not in self.datasets:
                        raise ValueError("Dataset not found")

                    model = self.models[model_id]
                    dataset = self.datasets[dataset_id]

                    if not model["trained"]:
                        raise RuntimeError("Model is not trained")

                    adapter = model["adapter"]
                    scores = adapter.score(dataset)
                    result = adapter.detect(dataset)

                    return {
                        "success": True,
                        "model_id": model_id,
                        "dataset_id": dataset_id,
                        "n_samples": len(scores),
                        "n_anomalies": int(np.sum(result.labels)),
                        "contamination_rate": float(np.mean(result.labels)),
                        "avg_score": float(np.mean([score.value for score in scores])),
                    }

                except Exception as e:
                    error_msg = f"Prediction failed: {str(e)}"
                    self._log_error(
                        error_msg, {"model_id": model_id, "dataset_id": dataset_id}
                    )

                    return {
                        "success": False,
                        "error": error_msg,
                        "model_id": model_id,
                        "dataset_id": dataset_id,
                    }

            def _get_circuit_breaker_state(self, circuit_key: str) -> str:
                """Get circuit breaker state."""
                if circuit_key not in self.circuit_breakers:
                    self.circuit_breakers[circuit_key] = {
                        "state": "closed",
                        "failure_count": 0,
                        "last_failure_time": None,
                        "failure_threshold": 5,
                        "timeout": 30,  # seconds
                    }

                circuit = self.circuit_breakers[circuit_key]

                if circuit["state"] == "open":
                    # Check if timeout has passed
                    if (
                        circuit["last_failure_time"]
                        and time.time() - circuit["last_failure_time"]
                        > circuit["timeout"]
                    ):
                        circuit["state"] = "half_open"

                return circuit["state"]

            def _record_circuit_failure(self, circuit_key: str):
                """Record circuit breaker failure."""
                circuit = self.circuit_breakers[circuit_key]
                circuit["failure_count"] += 1
                circuit["last_failure_time"] = time.time()

                if circuit["failure_count"] >= circuit["failure_threshold"]:
                    circuit["state"] = "open"

            def _record_circuit_success(self, circuit_key: str):
                """Record circuit breaker success."""
                circuit = self.circuit_breakers[circuit_key]
                circuit["failure_count"] = 0
                circuit["last_failure_time"] = None

            def _open_circuit_breaker(self, circuit_key: str):
                """Open circuit breaker."""
                circuit = self.circuit_breakers[circuit_key]
                circuit["state"] = "open"
                circuit["last_failure_time"] = time.time()

            def _close_circuit_breaker(self, circuit_key: str):
                """Close circuit breaker."""
                circuit = self.circuit_breakers[circuit_key]
                circuit["state"] = "closed"
                circuit["failure_count"] = 0
                circuit["last_failure_time"] = None

            def handle_invalid_data(
                self,
                data: pd.DataFrame | np.ndarray | list | dict,
                dataset_name: str = None,
            ) -> dict[str, Any]:
                """Handle potentially invalid data with validation and cleaning."""
                try:
                    # Convert input to DataFrame
                    if isinstance(data, dict):
                        df = pd.DataFrame([data])
                    elif isinstance(data, list):
                        if len(data) == 0:
                            return {
                                "success": False,
                                "error": "Empty data list",
                                "data": None,
                            }

                        if isinstance(data[0], dict):
                            df = pd.DataFrame(data)
                        else:
                            df = pd.DataFrame(data)
                    elif isinstance(data, np.ndarray):
                        if data.size == 0:
                            return {
                                "success": False,
                                "error": "Empty numpy array",
                                "data": None,
                            }

                        if data.ndim == 1:
                            df = pd.DataFrame(
                                data.reshape(-1, 1), columns=["feature_0"]
                            )
                        else:
                            df = pd.DataFrame(data)
                    elif isinstance(data, pd.DataFrame):
                        df = data.copy()
                    else:
                        return {
                            "success": False,
                            "error": f"Unsupported data type: {type(data)}",
                            "data": None,
                        }

                    # Validate and clean data
                    validation_result = self._validate_and_clean_data(df)

                    if not validation_result["valid"]:
                        return {
                            "success": False,
                            "error": f"Data validation failed: {validation_result['errors']}",
                            "data": None,
                            "issues": validation_result["issues"],
                        }

                    # Create dataset
                    cleaned_df = validation_result["cleaned_data"]
                    dataset_id = str(uuid.uuid4())
                    dataset_name = dataset_name or f"dataset_{dataset_id[:8]}"

                    self.datasets[dataset_id] = Dataset(
                        name=dataset_name, data=cleaned_df
                    )

                    return {
                        "success": True,
                        "dataset_id": dataset_id,
                        "dataset_name": dataset_name,
                        "n_samples": len(cleaned_df),
                        "n_features": len(cleaned_df.columns),
                        "cleaning_applied": validation_result["cleaning_applied"],
                        "warnings": validation_result.get("warnings", []),
                    }

                except Exception as e:
                    error_msg = f"Data handling failed: {str(e)}"
                    self._log_error(
                        error_msg,
                        {"data_type": type(data), "dataset_name": dataset_name},
                    )

                    return {"success": False, "error": error_msg, "data": None}

            def _validate_and_clean_data(self, df: pd.DataFrame) -> dict[str, Any]:
                """Validate and clean data."""
                issues = []
                warnings = []
                cleaning_applied = []
                cleaned_df = df.copy()

                # Check for empty DataFrame
                if len(cleaned_df) == 0:
                    return {
                        "valid": False,
                        "errors": ["Empty dataset"],
                        "issues": issues,
                    }

                # Check for minimum samples
                if len(cleaned_df) < 5:
                    issues.append(f"Very small dataset: {len(cleaned_df)} samples")
                    warnings.append("Small dataset may lead to unreliable results")

                # Handle missing values
                missing_values = cleaned_df.isnull().sum()
                if missing_values.any():
                    for col, missing_count in missing_values.items():
                        if missing_count > 0:
                            missing_ratio = missing_count / len(cleaned_df)

                            if missing_ratio > 0.5:
                                issues.append(
                                    f"Column '{col}' has {missing_ratio:.1%} missing values"
                                )
                                # Drop column with too many missing values
                                cleaned_df = cleaned_df.drop(columns=[col])
                                cleaning_applied.append(
                                    f"Dropped column '{col}' (too many missing values)"
                                )
                            elif missing_ratio > 0.1:
                                # Fill missing values
                                if cleaned_df[col].dtype in ["int64", "float64"]:
                                    cleaned_df[col] = cleaned_df[col].fillna(
                                        cleaned_df[col].mean()
                                    )
                                    cleaning_applied.append(
                                        f"Filled missing values in '{col}' with mean"
                                    )
                                else:
                                    cleaned_df[col] = cleaned_df[col].fillna(
                                        cleaned_df[col].mode().iloc[0]
                                        if not cleaned_df[col].mode().empty
                                        else "unknown"
                                    )
                                    cleaning_applied.append(
                                        f"Filled missing values in '{col}' with mode"
                                    )
                            else:
                                # Drop rows with few missing values
                                cleaned_df = cleaned_df.dropna(subset=[col])
                                cleaning_applied.append(
                                    f"Dropped rows with missing values in '{col}'"
                                )

                # Handle infinite values
                numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    inf_mask = np.isinf(cleaned_df[col])
                    if inf_mask.any():
                        # Replace infinite values with NaN, then fill with column statistics
                        cleaned_df.loc[inf_mask, col] = np.nan
                        cleaned_df[col] = cleaned_df[col].fillna(
                            cleaned_df[col].median()
                        )
                        cleaning_applied.append(
                            f"Replaced infinite values in '{col}' with median"
                        )

                # Convert non-numeric columns
                non_numeric_cols = cleaned_df.select_dtypes(exclude=[np.number]).columns
                for col in non_numeric_cols:
                    try:
                        # Try to convert to numeric
                        cleaned_df[col] = pd.to_numeric(
                            cleaned_df[col], errors="coerce"
                        )
                        if cleaned_df[col].isnull().all():
                            # If conversion failed for all values, drop the column
                            cleaned_df = cleaned_df.drop(columns=[col])
                            cleaning_applied.append(
                                f"Dropped non-numeric column '{col}'"
                            )
                        else:
                            # Fill any NaN values created by conversion
                            cleaned_df[col] = cleaned_df[col].fillna(
                                cleaned_df[col].median()
                            )
                            cleaning_applied.append(f"Converted '{col}' to numeric")
                    except Exception:
                        # Drop problematic column
                        cleaned_df = cleaned_df.drop(columns=[col])
                        cleaning_applied.append(f"Dropped problematic column '{col}'")

                # Check if any columns remain
                if len(cleaned_df.columns) == 0:
                    return {
                        "valid": False,
                        "errors": ["No valid columns after cleaning"],
                        "issues": issues,
                    }

                # Check for constant columns
                for col in cleaned_df.columns:
                    if cleaned_df[col].nunique() <= 1:
                        cleaned_df = cleaned_df.drop(columns=[col])
                        cleaning_applied.append(f"Dropped constant column '{col}'")

                # Final check
                if len(cleaned_df.columns) == 0:
                    return {
                        "valid": False,
                        "errors": ["No valid columns after removing constants"],
                        "issues": issues,
                    }

                if len(cleaned_df) == 0:
                    return {
                        "valid": False,
                        "errors": ["No valid rows after cleaning"],
                        "issues": issues,
                    }

                return {
                    "valid": True,
                    "errors": [],
                    "issues": issues,
                    "warnings": warnings,
                    "cleaned_data": cleaned_df,
                    "cleaning_applied": cleaning_applied,
                }

            def get_error_log(self, limit: int = None) -> list[dict[str, Any]]:
                """Get error log."""
                if limit:
                    return self.error_log[-limit:]
                return self.error_log

            def _log_error(self, message: str, context: dict[str, Any] = None):
                """Log error with context."""
                error_entry = {
                    "timestamp": time.time(),
                    "message": message,
                    "context": context or {},
                }
                self.error_log.append(error_entry)

                # Keep log size manageable
                if len(self.error_log) > 1000:
                    self.error_log = self.error_log[-500:]  # Keep last 500 entries

            def get_system_health(self) -> dict[str, Any]:
                """Get system health status."""
                current_time = time.time()

                # Count recent errors
                recent_errors = [
                    e for e in self.error_log if current_time - e["timestamp"] < 300
                ]  # Last 5 minutes

                # Check circuit breaker states
                open_circuits = [
                    k for k, v in self.circuit_breakers.items() if v["state"] == "open"
                ]

                # Check model health
                healthy_models = 0
                unhealthy_models = 0

                for _model_id, model_data in self.models.items():
                    if model_data["error_count"] > 10:
                        unhealthy_models += 1
                    else:
                        healthy_models += 1

                # Determine overall health
                if (
                    len(recent_errors) > 20
                    or len(open_circuits) > 5
                    or unhealthy_models > healthy_models
                ):
                    health_status = "unhealthy"
                elif (
                    len(recent_errors) > 10
                    or len(open_circuits) > 0
                    or unhealthy_models > 0
                ):
                    health_status = "degraded"
                else:
                    health_status = "healthy"

                return {
                    "status": health_status,
                    "total_models": len(self.models),
                    "healthy_models": healthy_models,
                    "unhealthy_models": unhealthy_models,
                    "total_datasets": len(self.datasets),
                    "recent_errors": len(recent_errors),
                    "open_circuits": len(open_circuits),
                    "circuit_details": open_circuits,
                    "error_rate": len(recent_errors) / 5
                    if len(recent_errors) > 0
                    else 0,  # errors per minute
                }

        return MockRobustSDK()

    def test_model_creation_with_fallback(self, mock_robust_sdk):
        """Test model creation with fallback configurations."""
        try:
            sdk = mock_robust_sdk

            # Test successful primary configuration
            primary_config = {
                "algorithm": "IsolationForest",
                "contamination": 0.1,
                "n_estimators": 50,
                "random_state": 42,
            }

            result = sdk.create_model_with_fallback(primary_config)

            assert result["status"] == "success"
            assert result["config_used"] == "primary"
            assert result["model_id"] is not None

            # Test fallback when primary fails
            invalid_primary = {
                "algorithm": "NonExistentAlgorithm",
                "contamination": 0.1,
            }

            fallback_configs = [
                {
                    "algorithm": "IsolationForest",
                    "contamination": 0.1,
                    "n_estimators": 30,
                },
                {
                    "algorithm": "LocalOutlierFactor",
                    "contamination": 0.1,
                    "n_neighbors": 20,
                    "novelty": True,
                },
            ]

            result = sdk.create_model_with_fallback(invalid_primary, fallback_configs)

            assert result["status"] == "success_with_fallback"
            assert result["config_used"].startswith("fallback_")
            assert result["model_id"] is not None
            assert "primary_error" in result

            # Test complete failure (all configs invalid)
            all_invalid = [{"algorithm": "Invalid1"}, {"algorithm": "Invalid2"}]

            result = sdk.create_model_with_fallback(
                {"algorithm": "Invalid0"}, all_invalid
            )

            assert result["status"] == "failed"
            assert result["model_id"] is None
            assert "primary_error" in result
            assert "fallback_errors" in result

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_training_with_retry_mechanism(self, mock_robust_sdk):
        """Test training with retry mechanism."""
        try:
            sdk = mock_robust_sdk

            # Create model
            config = {
                "algorithm": "IsolationForest",
                "contamination": 0.1,
                "n_estimators": 20,
                "random_state": 42,
            }

            result = sdk.create_model_with_fallback(config)
            model_id = result["model_id"]

            # Create valid dataset
            data = pd.DataFrame(
                {
                    "feature1": np.random.normal(0, 1, 100),
                    "feature2": np.random.normal(0, 1, 100),
                }
            )

            data_result = sdk.handle_invalid_data(data, "Valid Dataset")
            assert data_result["success"]
            dataset_id = data_result["dataset_id"]

            # Train with retry (should succeed on first attempt)
            training_result = sdk.train_with_retry(model_id, dataset_id)

            assert training_result["success"]
            assert training_result["model_id"] == model_id
            assert training_result["retries"] == 0  # Succeeded immediately
            assert "training_time" in training_result

            # Test retry with invalid model ID
            invalid_training = sdk.train_with_retry("invalid_model", dataset_id)

            assert not invalid_training["success"]
            assert "not found" in invalid_training["error"].lower()
            assert invalid_training["retries"] == 0

            # Test retry with invalid dataset ID
            invalid_dataset_training = sdk.train_with_retry(model_id, "invalid_dataset")

            assert not invalid_dataset_training["success"]
            assert "not found" in invalid_dataset_training["error"].lower()

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_invalid_data_handling_and_cleaning(self, mock_robust_sdk):
        """Test handling of various invalid data scenarios."""
        sdk = mock_robust_sdk

        # Test empty data
        empty_data = []
        result = sdk.handle_invalid_data(empty_data, "Empty Data")
        assert not result["success"]
        assert "empty" in result["error"].lower()

        # Test data with missing values
        missing_data = pd.DataFrame(
            {
                "feature1": [1, 2, np.nan, 4, 5],
                "feature2": [1, np.nan, 3, np.nan, 5],
                "feature3": [np.nan, np.nan, np.nan, np.nan, np.nan],  # All missing
            }
        )

        result = sdk.handle_invalid_data(missing_data, "Missing Data")
        assert result["success"]
        assert len(result["cleaning_applied"]) > 0
        assert (
            "feature3" not in sdk.datasets[result["dataset_id"]].data.columns
        )  # Should be dropped

        # Test data with infinite values
        infinite_data = pd.DataFrame(
            {"feature1": [1, 2, np.inf, 4, 5], "feature2": [1, 2, 3, -np.inf, 5]}
        )

        result = sdk.handle_invalid_data(infinite_data, "Infinite Data")
        assert result["success"]
        assert any("infinite" in clean for clean in result["cleaning_applied"])

        # Test mixed data types
        mixed_data = pd.DataFrame(
            {
                "numeric": [1, 2, 3, 4, 5],
                "text": ["a", "b", "c", "d", "e"],
                "convertible": ["1", "2", "3", "4", "5"],
            }
        )

        result = sdk.handle_invalid_data(mixed_data, "Mixed Data")
        assert result["success"]
        dataset = sdk.datasets[result["dataset_id"]]

        # Should have numeric and convertible columns
        assert len(dataset.data.columns) >= 2
        assert all(
            pd.api.types.is_numeric_dtype(dataset.data[col])
            for col in dataset.data.columns
        )

        # Test constant data
        constant_data = pd.DataFrame(
            {"constant": [1, 1, 1, 1, 1], "variable": [1, 2, 3, 4, 5]}
        )

        result = sdk.handle_invalid_data(constant_data, "Constant Data")
        assert result["success"]
        dataset = sdk.datasets[result["dataset_id"]]
        assert "constant" not in dataset.data.columns  # Should be dropped
        assert "variable" in dataset.data.columns

        # Test completely invalid data
        all_text_data = pd.DataFrame(
            {"text1": ["abc", "def", "ghi"], "text2": ["xyz", "uvw", "rst"]}
        )

        result = sdk.handle_invalid_data(all_text_data, "All Text Data")
        assert not result["success"]  # Should fail after cleaning

    def test_circuit_breaker_pattern(self, mock_robust_sdk):
        """Test circuit breaker pattern for prediction failures."""
        try:
            sdk = mock_robust_sdk

            # Create and train model
            config = {
                "algorithm": "IsolationForest",
                "contamination": 0.1,
                "n_estimators": 20,
                "random_state": 42,
            }

            model_result = sdk.create_model_with_fallback(config)
            model_id = model_result["model_id"]

            # Create dataset
            data = pd.DataFrame(
                {"x": np.random.normal(0, 1, 50), "y": np.random.normal(0, 1, 50)}
            )

            data_result = sdk.handle_invalid_data(data)
            dataset_id = data_result["dataset_id"]

            # Train model
            training_result = sdk.train_with_retry(model_id, dataset_id)
            assert training_result["success"]

            # Test successful prediction (circuit closed)
            prediction_result = sdk.predict_with_circuit_breaker(model_id, dataset_id)
            assert prediction_result["success"]
            assert prediction_result["circuit_state"] == "closed"

            # Test prediction with invalid dataset (simulate failures)
            invalid_dataset_id = "invalid_dataset"

            # Generate multiple failures to trigger circuit breaker
            failure_count = 0
            for _i in range(10):  # More than failure threshold
                result = sdk.predict_with_circuit_breaker(model_id, invalid_dataset_id)
                if not result["success"]:
                    failure_count += 1

                # Check if circuit opens
                if result.get("circuit_state") == "open":
                    break

            # Verify circuit breaker opened
            assert failure_count >= 5  # Should have triggered circuit breaker

            # Test that subsequent requests are immediately rejected
            immediate_result = sdk.predict_with_circuit_breaker(
                model_id, invalid_dataset_id
            )
            assert not immediate_result["success"]
            assert immediate_result["circuit_state"] == "open"
            assert "circuit breaker is open" in immediate_result["error"].lower()

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_error_logging_and_monitoring(self, mock_robust_sdk):
        """Test error logging and system monitoring."""
        try:
            sdk = mock_robust_sdk

            # Generate some errors
            invalid_configs = [
                {"algorithm": "InvalidAlgo1"},
                {"algorithm": "InvalidAlgo2"},
                {"algorithm": "InvalidAlgo3"},
            ]

            for config in invalid_configs:
                try:
                    sdk.create_model_with_fallback(config)
                except Exception:
                    pass

            # Check error log
            error_log = sdk.get_error_log()
            assert len(error_log) >= 3  # Should have logged the failures

            for error in error_log:
                assert "timestamp" in error
                assert "message" in error
                assert "context" in error

            # Test limited error log retrieval
            limited_log = sdk.get_error_log(limit=2)
            assert len(limited_log) <= 2

            # Check system health
            health = sdk.get_system_health()

            assert "status" in health
            assert health["status"] in ["healthy", "degraded", "unhealthy"]
            assert "total_models" in health
            assert "recent_errors" in health
            assert "error_rate" in health
            assert health["error_rate"] >= 0

            # Health status should reflect recent errors
            if health["recent_errors"] > 0:
                assert health["status"] in ["degraded", "unhealthy"]

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_concurrent_error_handling(self, mock_robust_sdk):
        """Test error handling under concurrent operations."""
        try:
            sdk = mock_robust_sdk

            # Create model
            config = {
                "algorithm": "IsolationForest",
                "contamination": 0.1,
                "n_estimators": 15,
                "random_state": 42,
            }

            model_result = sdk.create_model_with_fallback(config)
            model_id = model_result["model_id"]

            # Create dataset
            data = pd.DataFrame({"feature": np.random.normal(0, 1, 100)})

            data_result = sdk.handle_invalid_data(data)
            dataset_id = data_result["dataset_id"]

            # Train model
            training_result = sdk.train_with_retry(model_id, dataset_id)
            assert training_result["success"]

            # Concurrent operations with error scenarios
            results = []
            errors = []

            def worker(worker_id):
                try:
                    if worker_id % 3 == 0:
                        # Valid operation
                        result = sdk.predict_with_circuit_breaker(model_id, dataset_id)
                        results.append(result)
                    elif worker_id % 3 == 1:
                        # Invalid dataset
                        result = sdk.predict_with_circuit_breaker(
                            model_id, "invalid_dataset"
                        )
                        results.append(result)
                    else:
                        # Invalid model
                        result = sdk.predict_with_circuit_breaker(
                            "invalid_model", dataset_id
                        )
                        results.append(result)
                except Exception as e:
                    errors.append(str(e))

            # Run concurrent operations
            threads = []
            for i in range(12):  # Mix of valid and invalid operations
                thread = threading.Thread(target=worker, args=(i,))
                threads.append(thread)
                thread.start()

            # Wait for completion
            for thread in threads:
                thread.join()

            # Verify results
            assert len(results) + len(errors) == 12

            # Should have some successful and some failed operations
            successful_ops = [r for r in results if r.get("success", False)]
            failed_ops = [r for r in results if not r.get("success", True)]

            assert len(successful_ops) > 0  # Some operations should succeed
            assert len(failed_ops) > 0  # Some operations should fail

            # Check that error handling was thread-safe
            error_log = sdk.get_error_log()
            assert len(error_log) >= len(failed_ops)

            # System should still be functional
            health = sdk.get_system_health()
            assert health["status"] in ["healthy", "degraded", "unhealthy"]

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_graceful_degradation(self, mock_robust_sdk):
        """Test graceful degradation under adverse conditions."""
        try:
            sdk = mock_robust_sdk

            # Create multiple models with different reliability
            models = []

            # Reliable model
            reliable_config = {
                "algorithm": "IsolationForest",
                "contamination": 0.1,
                "n_estimators": 10,
                "random_state": 42,
            }

            reliable_result = sdk.create_model_with_fallback(reliable_config)
            models.append(reliable_result["model_id"])

            # Create datasets
            good_data = pd.DataFrame(
                {
                    "feature1": np.random.normal(0, 1, 50),
                    "feature2": np.random.normal(0, 1, 50),
                }
            )

            good_data_result = sdk.handle_invalid_data(good_data, "Good Data")
            good_dataset_id = good_data_result["dataset_id"]

            # Train models
            for model_id in models:
                training_result = sdk.train_with_retry(model_id, good_dataset_id)
                assert training_result["success"]

            # Test system under stress
            stress_results = []

            # Generate load with mixed valid/invalid requests
            for i in range(50):
                if i % 5 == 0:
                    # Valid request
                    result = sdk.predict_with_circuit_breaker(
                        models[0], good_dataset_id
                    )
                elif i % 5 == 1:
                    # Invalid dataset
                    result = sdk.predict_with_circuit_breaker(
                        models[0], "invalid_dataset"
                    )
                elif i % 5 == 2:
                    # Invalid model
                    result = sdk.predict_with_circuit_breaker(
                        "invalid_model", good_dataset_id
                    )
                else:
                    # Data handling stress
                    problematic_data = pd.DataFrame(
                        {"bad_feature": [np.nan, np.inf, "text", -np.inf, None]}
                    )
                    result = sdk.handle_invalid_data(
                        problematic_data, f"Stress Data {i}"
                    )

                stress_results.append(result)

            # System should still be responsive
            health = sdk.get_system_health()
            assert health["status"] in ["healthy", "degraded", "unhealthy"]

            # At least some operations should have succeeded
            successful_results = [r for r in stress_results if r.get("success", False)]
            assert len(successful_results) > 0

            # Error rate should be manageable
            assert health["error_rate"] < 20  # Less than 20 errors per minute

            # Test recovery - valid operations should still work
            recovery_result = sdk.predict_with_circuit_breaker(
                models[0], good_dataset_id
            )

            # Should either succeed or have clear error message
            if not recovery_result["success"]:
                assert "error" in recovery_result
                assert len(recovery_result["error"]) > 0

        except ImportError:
            pytest.skip("scikit-learn not available")
