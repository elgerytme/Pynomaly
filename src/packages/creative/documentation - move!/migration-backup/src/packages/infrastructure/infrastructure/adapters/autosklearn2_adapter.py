"""Auto-sklearn2 adapter for automated anomaly detection.

This module provides integration with auto-sklearn2 for automated
machine learning-based anomaly detection.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from monorepo.domain.entities import Detector
from monorepo.domain.exceptions import DetectorError, TrainingError
from monorepo.shared.protocols import DetectorProtocol

# Optional auto-sklearn2 import
try:
    from autosklearn2.classification import AutoSklearnClassifier

    AUTOSKLEARN2_AVAILABLE = True
except ImportError:
    AUTOSKLEARN2_AVAILABLE = False

logger = logging.getLogger(__name__)


class AutoSklearn2Adapter(DetectorProtocol):
    """Adapter for auto-sklearn2 automated anomaly detection.

    This adapter uses auto-sklearn2's automated machine learning capabilities
    to automatically select and optimize anomaly detection models.
    """

    def __init__(self):
        """Initialize the auto-sklearn2 adapter."""
        if not AUTOSKLEARN2_AVAILABLE:
            raise ImportError(
                "auto-sklearn2 is not available. Install with: "
                "pip install pynomaly[automl] or pip install auto-sklearn2"
            )

        self.name = "AutoSklearn2"
        self.description = "Automated anomaly detection using auto-sklearn2"
        self.version = "1.0.0"
        self.algorithm_type = "ensemble"

        # Store fitted models
        self._models: dict[str, Any] = {}
        self._preprocessors: dict[str, Any] = {}
        self._thresholds: dict[str, float] = {}

        logger.info("Auto-sklearn2 adapter initialized")

    @property
    def is_available(self) -> bool:
        """Check if auto-sklearn2 is available."""
        return AUTOSKLEARN2_AVAILABLE

    @property
    def supported_algorithms(self) -> list[str]:
        """Get supported algorithm names."""
        return ["AutoSklearn2OneClass", "AutoSklearn2Outlier", "AutoSklearn2Ensemble"]

    def get_algorithm_info(self, algorithm: str) -> dict[str, Any]:
        """Get information about a specific algorithm."""
        info_map = {
            "AutoSklearn2OneClass": {
                "name": "AutoSklearn2OneClass",
                "description": "One-class classification using auto-sklearn2",
                "type": "supervised",
                "parameters": {
                    "time_left_for_this_task": {
                        "type": "int",
                        "default": 300,
                        "description": "Time limit for AutoML optimization (seconds)",
                    },
                    "per_run_time_limit": {
                        "type": "int",
                        "default": 30,
                        "description": "Time limit per individual model run (seconds)",
                    },
                    "ensemble_size": {
                        "type": "int",
                        "default": 10,
                        "description": "Number of models in final ensemble",
                    },
                    "memory_limit": {
                        "type": "int",
                        "default": 3072,
                        "description": "Memory limit in MB",
                    },
                },
            },
            "AutoSklearn2Outlier": {
                "name": "AutoSklearn2Outlier",
                "description": "Outlier detection using auto-sklearn2 classification",
                "type": "semi-supervised",
                "parameters": {
                    "time_left_for_this_task": {
                        "type": "int",
                        "default": 300,
                        "description": "Time limit for AutoML optimization (seconds)",
                    },
                    "contamination": {
                        "type": "float",
                        "default": 0.1,
                        "description": "Expected contamination ratio",
                    },
                    "validation_strategy": {
                        "type": "str",
                        "default": "holdout",
                        "description": "Validation strategy for model selection",
                    },
                },
            },
            "AutoSklearn2Ensemble": {
                "name": "AutoSklearn2Ensemble",
                "description": "Ensemble of auto-sklearn2 models",
                "type": "ensemble",
                "parameters": {
                    "n_estimators": {
                        "type": "int",
                        "default": 3,
                        "description": "Number of auto-sklearn2 models in ensemble",
                    },
                    "time_per_model": {
                        "type": "int",
                        "default": 100,
                        "description": "Time limit per model (seconds)",
                    },
                    "ensemble_method": {
                        "type": "str",
                        "default": "voting",
                        "description": "Ensemble combination method",
                    },
                },
            },
        }

        return info_map.get(algorithm, {})

    def validate_parameters(self, algorithm: str, parameters: dict[str, Any]) -> bool:
        """Validate algorithm parameters."""
        try:
            algorithm_info = self.get_algorithm_info(algorithm)
            if not algorithm_info:
                return False

            expected_params = algorithm_info.get("parameters", {})

            for param_name, param_value in parameters.items():
                if param_name not in expected_params:
                    logger.warning(f"Unknown parameter: {param_name}")
                    continue

                param_info = expected_params[param_name]
                param_type = param_info["type"]

                # Type validation
                if param_type == "int" and not isinstance(param_value, int):
                    return False
                elif param_type == "float" and not isinstance(param_value, int | float):
                    return False
                elif param_type == "str" and not isinstance(param_value, str):
                    return False

            return True

        except Exception as e:
            logger.error(f"Parameter validation error: {str(e)}")
            return False

    def train(
        self, detector: Detector, X: np.ndarray, y: np.ndarray | None = None
    ) -> bool:
        """Train the auto-sklearn2 model.

        Args:
            detector: Detector configuration
            X: Training features
            y: Training labels (optional, for semi-supervised learning)

        Returns:
            True if training successful, False otherwise
        """
        try:
            if not self.is_available:
                raise DetectorError("auto-sklearn2 is not available")

            algorithm = detector.algorithm
            parameters = detector.hyperparameters or {}

            logger.info(f"Training {algorithm} with auto-sklearn2")
            start_time = time.time()

            # Validate parameters
            if not self.validate_parameters(algorithm, parameters):
                raise TrainingError(f"Invalid parameters for {algorithm}")

            # Prepare training approach based on algorithm
            if algorithm == "AutoSklearn2OneClass":
                model = self._train_oneclass(X, parameters)
            elif algorithm == "AutoSklearn2Outlier":
                model = self._train_outlier(X, y, parameters)
            elif algorithm == "AutoSklearn2Ensemble":
                model = self._train_ensemble(X, y, parameters)
            else:
                raise TrainingError(f"Unsupported algorithm: {algorithm}")

            # Store model and metadata
            self._models[detector.id] = model
            self._preprocessors[detector.id] = (
                None  # auto-sklearn2 handles preprocessing
            )

            # Calculate threshold for anomaly detection
            scores = self._compute_scores(model, X, algorithm)
            contamination = parameters.get("contamination", 0.1)
            threshold = np.percentile(scores, (1 - contamination) * 100)
            self._thresholds[detector.id] = threshold

            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f}s")

            return True

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return False

    def _train_oneclass(
        self, X: np.ndarray, parameters: dict[str, Any]
    ) -> AutoSklearnClassifier:
        """Train one-class classifier using auto-sklearn2."""
        # Convert to binary classification problem
        # Create artificial anomalies for training
        n_samples = len(X)
        contamination = parameters.get("contamination", 0.1)
        n_anomalies = int(n_samples * contamination)

        # Generate synthetic anomalies by adding noise
        anomalies = X + np.random.normal(0, np.std(X) * 2, X.shape)[:n_anomalies]

        # Combine normal and anomalous samples
        X_combined = np.vstack([X, anomalies])
        y_combined = np.hstack([np.zeros(n_samples), np.ones(n_anomalies)])

        # Configure auto-sklearn2
        time_limit = parameters.get("time_left_for_this_task", 300)
        per_run_limit = parameters.get("per_run_time_limit", 30)
        memory_limit = parameters.get("memory_limit", 3072)
        ensemble_size = parameters.get("ensemble_size", 10)

        classifier = AutoSklearnClassifier(
            time_left_for_this_task=time_limit,
            per_run_time_limit=per_run_limit,
            memory_limit=memory_limit,
            ensemble_size=ensemble_size,
            include={
                "classifier": [
                    "random_forest",
                    "extra_trees",
                    "gradient_boosting",
                    "adaboost",
                    "decision_tree",
                    "k_nearest_neighbors",
                ],
                "feature_preprocessor": [
                    "select_percentile_classification",
                    "pca",
                    "truncatedSVD",
                ],
            },
            resampling_strategy="holdout",
            resampling_strategy_arguments={"train_size": 0.8},
        )

        # Train the model
        classifier.fit(X_combined, y_combined)

        return classifier

    def _train_outlier(
        self, X: np.ndarray, y: np.ndarray | None, parameters: dict[str, Any]
    ) -> AutoSklearnClassifier:
        """Train outlier detector using auto-sklearn2."""
        if y is None:
            # Use one-class approach if no labels provided
            return self._train_oneclass(X, parameters)

        # Use provided labels for supervised training
        time_limit = parameters.get("time_left_for_this_task", 300)
        validation_strategy = parameters.get("validation_strategy", "holdout")

        classifier = AutoSklearnClassifier(
            time_left_for_this_task=time_limit,
            resampling_strategy=validation_strategy,
            include={
                "classifier": [
                    "random_forest",
                    "extra_trees",
                    "gradient_boosting",
                    "svm",
                    "libsvm_svc",
                    "decision_tree",
                ],
                "feature_preprocessor": [
                    "select_percentile_classification",
                    "pca",
                    "polynomial",
                ],
            },
        )

        classifier.fit(X, y)
        return classifier

    def _train_ensemble(
        self, X: np.ndarray, y: np.ndarray | None, parameters: dict[str, Any]
    ) -> list[AutoSklearnClassifier]:
        """Train ensemble of auto-sklearn2 models."""
        n_estimators = parameters.get("n_estimators", 3)
        time_per_model = parameters.get("time_per_model", 100)
        parameters.get("ensemble_method", "voting")

        models = []

        for i in range(n_estimators):
            logger.info(f"Training ensemble model {i + 1}/{n_estimators}")

            # Use different random states and configurations for diversity
            model_params = parameters.copy()
            model_params["time_left_for_this_task"] = time_per_model

            if y is None:
                model = self._train_oneclass(X, model_params)
            else:
                model = self._train_outlier(X, y, model_params)

            models.append(model)

        return models

    def _compute_scores(
        self,
        model: AutoSklearnClassifier | list[AutoSklearnClassifier],
        X: np.ndarray,
        algorithm: str,
    ) -> np.ndarray:
        """Compute anomaly scores."""
        if algorithm == "AutoSklearn2Ensemble":
            # Ensemble scoring
            scores_list = []
            for m in model:
                try:
                    prob = m.predict_proba(X)
                    if prob.shape[1] == 2:  # Binary classification
                        scores_list.append(prob[:, 1])  # Probability of being anomaly
                    else:
                        scores_list.append(np.max(prob, axis=1))
                except Exception as e:
                    logger.warning(f"Error computing scores for ensemble member: {e}")
                    continue

            if scores_list:
                return np.mean(scores_list, axis=0)
            else:
                return np.zeros(len(X))
        else:
            # Single model scoring
            try:
                prob = model.predict_proba(X)
                if prob.shape[1] == 2:
                    return prob[:, 1]  # Probability of being anomaly
                else:
                    return np.max(prob, axis=1)
            except Exception as e:
                logger.error(f"Error computing scores: {e}")
                return np.zeros(len(X))

    def predict(
        self, detector: Detector, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict anomalies using trained model.

        Args:
            detector: Trained detector
            X: Features to predict on

        Returns:
            Tuple of (predictions, scores)
        """
        try:
            if detector.id not in self._models:
                raise DetectorError(f"Detector {detector.id} not trained")

            model = self._models[detector.id]
            threshold = self._thresholds[detector.id]

            # Compute anomaly scores
            scores = self._compute_scores(model, X, detector.algorithm)

            # Convert scores to binary predictions
            predictions = (scores > threshold).astype(int)

            logger.info(
                f"Predicted {np.sum(predictions)} anomalies out of {len(X)} samples"
            )

            return predictions, scores

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            # Return zero predictions on error
            return np.zeros(len(X), dtype=int), np.zeros(len(X))

    def get_feature_importance(self, detector: Detector) -> np.ndarray | None:
        """Get feature importance from trained model.

        Args:
            detector: Trained detector

        Returns:
            Feature importance array or None if not available
        """
        try:
            if detector.id not in self._models:
                return None

            model = self._models[detector.id]

            if detector.algorithm == "AutoSklearn2Ensemble":
                # Average feature importance across ensemble
                importance_list = []
                for m in model:
                    try:
                        # Try to get feature importance from underlying models
                        if hasattr(m, "get_models_with_weights"):
                            models_weights = m.get_models_with_weights()
                            for model_weight_tuple in models_weights:
                                base_model = model_weight_tuple[0]
                                if hasattr(base_model, "feature_importances_"):
                                    importance_list.append(
                                        base_model.feature_importances_
                                    )
                    except Exception:
                        continue

                if importance_list:
                    return np.mean(importance_list, axis=0)
            else:
                # Single model feature importance
                try:
                    if hasattr(model, "get_models_with_weights"):
                        models_weights = model.get_models_with_weights()
                        for model_weight_tuple in models_weights:
                            base_model = model_weight_tuple[0]
                            if hasattr(base_model, "feature_importances_"):
                                return base_model.feature_importances_
                except Exception:
                    pass

            return None

        except Exception as e:
            logger.error(f"Feature importance extraction failed: {str(e)}")
            return None

    def save_model(self, detector: Detector, path: str) -> bool:
        """Save trained model to disk.

        Args:
            detector: Trained detector
            path: Path to save model

        Returns:
            True if successful, False otherwise
        """
        try:
            import pickle

            if detector.id not in self._models:
                return False

            model_data = {
                "model": self._models[detector.id],
                "threshold": self._thresholds[detector.id],
                "algorithm": detector.algorithm,
                "parameters": detector.hyperparameters,
            }

            with open(path, "wb") as f:
                pickle.dump(model_data, f)

            logger.info(f"Model saved to {path}")
            return True

        except Exception as e:
            logger.error(f"Model saving failed: {str(e)}")
            return False

    def load_model(self, detector: Detector, path: str) -> bool:
        """Load trained model from disk.

        Args:
            detector: Detector to load model into
            path: Path to load model from

        Returns:
            True if successful, False otherwise
        """
        try:
            import pickle

            with open(path, "rb") as f:
                model_data = pickle.load(f)

            self._models[detector.id] = model_data["model"]
            self._thresholds[detector.id] = model_data["threshold"]

            logger.info(f"Model loaded from {path}")
            return True

        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            return False

    def cleanup(self, detector: Detector) -> None:
        """Clean up resources for a detector.

        Args:
            detector: Detector to clean up
        """
        try:
            if detector.id in self._models:
                del self._models[detector.id]
            if detector.id in self._preprocessors:
                del self._preprocessors[detector.id]
            if detector.id in self._thresholds:
                del self._thresholds[detector.id]

            logger.info(f"Cleaned up resources for detector {detector.id}")

        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")

    def get_model_info(self, detector: Detector) -> dict[str, Any]:
        """Get information about trained model.

        Args:
            detector: Trained detector

        Returns:
            Model information dictionary
        """
        try:
            if detector.id not in self._models:
                return {}

            model = self._models[detector.id]
            info = {
                "adapter": self.name,
                "algorithm": detector.algorithm,
                "threshold": self._thresholds.get(detector.id, 0.0),
                "auto_sklearn2_version": "1.0.0+",
                "ensemble_size": 0,
                "training_time": None,
            }

            # Try to get model-specific information
            if detector.algorithm == "AutoSklearn2Ensemble":
                info["ensemble_size"] = len(model)
                info["models"] = [type(m).__name__ for m in model]
            else:
                try:
                    if hasattr(model, "get_models_with_weights"):
                        models_weights = model.get_models_with_weights()
                        info["ensemble_size"] = len(models_weights)
                        info["models"] = [type(m[0]).__name__ for m in models_weights]
                except Exception:
                    pass

            return info

        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {}
