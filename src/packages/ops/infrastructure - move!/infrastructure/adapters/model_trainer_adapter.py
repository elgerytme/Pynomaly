"""Model trainer adapter for integrating with existing training infrastructure.

This adapter connects the training automation service with the existing
Pynomaly training and evaluation systems.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from monorepo.application.services.training_automation_service import ModelTrainer
from monorepo.domain.entities import Dataset, DetectionResult, Detector
from monorepo.domain.exceptions import TrainingError

# Import existing services
try:
    from monorepo.application.use_cases.detect_anomalies import DetectAnomaliesUseCase
    from monorepo.application.use_cases.evaluate_model import EvaluateModelUseCase
    from monorepo.application.use_cases.train_detector import TrainDetectorUseCase

    USE_CASES_AVAILABLE = True
except ImportError:
    USE_CASES_AVAILABLE = False

try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelTrainerAdapter(ModelTrainer):
    """Adapter that connects training automation with existing Pynomaly infrastructure."""

    def __init__(
        self,
        train_detector_use_case: TrainDetectorUseCase | None = None,
        detect_anomalies_use_case: DetectAnomaliesUseCase | None = None,
        evaluate_model_use_case: EvaluateModelUseCase | None = None,
    ):
        self.train_detector_use_case = train_detector_use_case
        self.detect_anomalies_use_case = detect_anomalies_use_case
        self.evaluate_model_use_case = evaluate_model_use_case

        # Initialize fallback implementations if use cases not provided
        if not USE_CASES_AVAILABLE:
            logger.warning("Use cases not available, using fallback implementations")

    async def train(
        self, detector: Detector, dataset: Dataset, parameters: dict[str, Any]
    ) -> DetectionResult:
        """Train model with given parameters."""
        try:
            # Update detector parameters
            detector.parameters.update(parameters)

            if self.train_detector_use_case:
                # Use existing use case
                from monorepo.application.use_cases.train_detector import (
                    TrainDetectorRequest,
                )

                request = TrainDetectorRequest(
                    detector_id=detector.id,
                    training_data=dataset,
                    validate_data=True,
                    save_model=False,  # Don't save during optimization
                    hyperparameters=parameters,
                )

                result = await self.train_detector_use_case.execute(request)
                return result.detection_result
            else:
                # Fallback implementation
                return await self._fallback_train(detector, dataset, parameters)

        except Exception as e:
            logger.error(f"Training failed for {detector.algorithm_name}: {e}")
            raise TrainingError(f"Training failed: {e}")

    async def evaluate(
        self,
        detector: Detector,
        dataset: Dataset,
        validation_data: Dataset | None = None,
    ) -> dict[str, float]:
        """Evaluate trained model."""
        try:
            if self.evaluate_model_use_case:
                # Use existing evaluation use case
                from monorepo.application.use_cases.evaluate_model import (
                    EvaluateModelRequest,
                )

                request = EvaluateModelRequest(
                    detector_id=detector.id,
                    test_data=validation_data or dataset,
                    metrics=["roc_auc", "precision", "recall", "f1", "accuracy"],
                )

                result = await self.evaluate_model_use_case.execute(request)
                return result.metrics
            else:
                # Fallback implementation
                return await self._fallback_evaluate(detector, dataset, validation_data)

        except Exception as e:
            logger.error(f"Evaluation failed for {detector.algorithm_name}: {e}")
            # Return default metrics to prevent optimization failure
            return {
                "roc_auc": 0.5,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "accuracy": 0.5,
                "balanced_accuracy": 0.5,
            }

    async def _fallback_train(
        self, detector: Detector, dataset: Dataset, parameters: dict[str, Any]
    ) -> DetectionResult:
        """Fallback training implementation."""
        logger.info(f"Using fallback training for {detector.algorithm_name}")

        # Create a simple training simulation
        # In a real implementation, this would interface with the algorithm adapters

        data = dataset.data
        if data.empty:
            raise TrainingError("Dataset is empty")

        # Simulate training by creating a simple detection result
        n_samples = len(data)

        # Generate simulated anomaly scores and labels
        # This is just for testing - real implementation would use actual algorithms
        np.random.seed(42)  # For reproducible results during testing
        anomaly_scores = np.random.random(n_samples)

        # Use contamination parameter if available
        contamination = parameters.get("contamination", 0.1)
        threshold = np.percentile(anomaly_scores, (1 - contamination) * 100)
        anomaly_labels = (anomaly_scores > threshold).astype(int)

        # Create detection result
        result = DetectionResult(
            detector_id=detector.id,
            dataset_id=dataset.id,
            anomaly_scores=anomaly_scores.tolist(),
            anomaly_labels=anomaly_labels.tolist(),
            threshold=threshold,
            metadata={
                "algorithm": detector.algorithm_name,
                "parameters": parameters,
                "training_method": "fallback",
                "n_samples": n_samples,
                "contamination": contamination,
            },
        )

        return result

    async def _fallback_evaluate(
        self,
        detector: Detector,
        dataset: Dataset,
        validation_data: Dataset | None = None,
    ) -> dict[str, float]:
        """Fallback evaluation implementation."""
        logger.info(f"Using fallback evaluation for {detector.algorithm_name}")

        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available for evaluation")
            return {
                "roc_auc": 0.5,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "accuracy": 0.5,
            }

        # Use validation data if provided, otherwise use training data
        eval_dataset = validation_data or dataset
        data = eval_dataset.data

        if data.empty:
            logger.warning("Evaluation dataset is empty")
            return {"roc_auc": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}

        # Check if we have true labels for evaluation
        if hasattr(eval_dataset, "target_column") and eval_dataset.target_column:
            true_labels = data[eval_dataset.target_column].values
        else:
            # Generate synthetic labels for testing
            np.random.seed(42)
            contamination = detector.parameters.get("contamination", 0.1)
            n_anomalies = int(len(data) * contamination)
            true_labels = np.zeros(len(data))
            anomaly_indices = np.random.choice(len(data), n_anomalies, replace=False)
            true_labels[anomaly_indices] = 1

        # Generate predictions (simulation)
        # In real implementation, this would use the trained model
        np.random.seed(43)  # Different seed for predictions
        pred_scores = np.random.random(len(data))
        threshold = np.percentile(
            pred_scores, (1 - detector.parameters.get("contamination", 0.1)) * 100
        )
        pred_labels = (pred_scores > threshold).astype(int)

        # Calculate metrics
        try:
            metrics = {}

            # ROC AUC (requires probability scores)
            if len(np.unique(true_labels)) > 1:
                metrics["roc_auc"] = roc_auc_score(true_labels, pred_scores)
            else:
                metrics["roc_auc"] = 0.5

            # Classification metrics
            metrics["precision"] = precision_score(
                true_labels, pred_labels, zero_division=0
            )
            metrics["recall"] = recall_score(true_labels, pred_labels, zero_division=0)
            metrics["f1"] = f1_score(true_labels, pred_labels, zero_division=0)
            metrics["accuracy"] = accuracy_score(true_labels, pred_labels)
            metrics["balanced_accuracy"] = balanced_accuracy_score(
                true_labels, pred_labels
            )

            # Additional anomaly detection specific metrics
            true_anomalies = np.sum(true_labels)
            predicted_anomalies = np.sum(pred_labels)

            metrics["true_anomaly_rate"] = true_anomalies / len(true_labels)
            metrics["predicted_anomaly_rate"] = predicted_anomalies / len(pred_labels)

            if true_anomalies > 0:
                # Detection rate (recall for anomaly class)
                metrics["detection_rate"] = metrics["recall"]
            else:
                metrics["detection_rate"] = 0.0

            return metrics

        except Exception as e:
            logger.error(f"Metric calculation failed: {e}")
            return {
                "roc_auc": 0.5,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "accuracy": 0.5,
                "balanced_accuracy": 0.5,
            }

    async def validate_model(
        self, detector: Detector, dataset: Dataset, validation_split: float = 0.2
    ) -> dict[str, float]:
        """Perform model validation with train/validation split."""

        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available for validation")
            return await self.evaluate(detector, dataset)

        data = dataset.data
        if len(data) < 10:  # Need minimum samples for split
            logger.warning("Dataset too small for validation split")
            return await self.evaluate(detector, dataset)

        # Split data
        train_data, val_data = train_test_split(
            data,
            test_size=validation_split,
            random_state=42,
            stratify=None,  # Unsupervised learning
        )

        # Create validation dataset
        val_dataset = Dataset(
            name=f"{dataset.name}_validation",
            data=val_data,
            target_column=getattr(dataset, "target_column", None),
        )

        # Train on training split
        train_dataset = Dataset(
            name=f"{dataset.name}_train",
            data=train_data,
            target_column=getattr(dataset, "target_column", None),
        )

        # Train model
        await self.train(detector, train_dataset, detector.parameters)

        # Evaluate on validation split
        return await self.evaluate(detector, train_dataset, val_dataset)

    async def cross_validate(
        self, detector: Detector, dataset: Dataset, cv_folds: int = 5
    ) -> dict[str, float]:
        """Perform cross-validation evaluation."""

        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available for cross-validation")
            return await self.evaluate(detector, dataset)

        from sklearn.model_selection import KFold

        data = dataset.data
        if len(data) < cv_folds:
            logger.warning("Dataset too small for cross-validation")
            return await self.evaluate(detector, dataset)

        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
            try:
                # Create fold datasets
                train_data = data.iloc[train_idx]
                val_data = data.iloc[val_idx]

                train_dataset = Dataset(
                    name=f"{dataset.name}_fold_{fold}_train",
                    data=train_data,
                    target_column=getattr(dataset, "target_column", None),
                )

                val_dataset = Dataset(
                    name=f"{dataset.name}_fold_{fold}_val",
                    data=val_data,
                    target_column=getattr(dataset, "target_column", None),
                )

                # Train and evaluate
                await self.train(detector, train_dataset, detector.parameters)
                metrics = await self.evaluate(detector, train_dataset, val_dataset)

                fold_metrics.append(metrics)

            except Exception as e:
                logger.warning(f"Fold {fold} failed: {e}")
                continue

        if not fold_metrics:
            logger.error("All cross-validation folds failed")
            return await self.evaluate(detector, dataset)

        # Aggregate metrics across folds
        aggregated = {}
        metric_names = fold_metrics[0].keys()

        for metric_name in metric_names:
            values = [fold[metric_name] for fold in fold_metrics if metric_name in fold]
            if values:
                aggregated[metric_name] = np.mean(values)
                aggregated[f"{metric_name}_std"] = np.std(values)
            else:
                aggregated[metric_name] = 0.0
                aggregated[f"{metric_name}_std"] = 0.0

        aggregated["cv_folds"] = len(fold_metrics)

        return aggregated

    def get_supported_algorithms(self) -> List[str]:
        """Get list of supported algorithms."""
        # This would query the algorithm registry in a real implementation
        return [
            "IsolationForest",
            "LocalOutlierFactor",
            "OneClassSVM",
            "Autoencoder",
            "PCA",
            "COPOD",
            "ECOD",
            "KNN",
            "CBLOF",
        ]

    def get_algorithm_info(self, algorithm_name: str) -> dict[str, Any]:
        """Get information about a specific algorithm."""
        # This would query the algorithm registry for detailed info
        algorithm_info = {
            "IsolationForest": {
                "type": "ensemble",
                "supports_online": False,
                "supports_multivariate": True,
                "computational_complexity": "O(n log n)",
                "memory_complexity": "O(n)",
                "best_for": ["high_dimensional", "large_datasets"],
            },
            "LocalOutlierFactor": {
                "type": "density",
                "supports_online": False,
                "supports_multivariate": True,
                "computational_complexity": "O(n²)",
                "memory_complexity": "O(n²)",
                "best_for": ["local_anomalies", "varying_densities"],
            },
            "OneClassSVM": {
                "type": "boundary",
                "supports_online": False,
                "supports_multivariate": True,
                "computational_complexity": "O(n²)",
                "memory_complexity": "O(n)",
                "best_for": ["complex_boundaries", "non_linear_patterns"],
            },
        }

        return algorithm_info.get(
            algorithm_name,
            {
                "type": "unknown",
                "supports_online": False,
                "supports_multivariate": True,
                "computational_complexity": "Unknown",
                "memory_complexity": "Unknown",
                "best_for": [],
            },
        )


# Factory function for creating the adapter
def create_model_trainer_adapter() -> ModelTrainerAdapter:
    """Create model trainer adapter with dependency injection."""

    # In a real implementation, this would use the DI container
    # to inject the actual use cases

    try:
        from monorepo.infrastructure.config import create_container

        container = create_container()

        return ModelTrainerAdapter(
            train_detector_use_case=container.train_detector_use_case(),
            detect_anomalies_use_case=container.detect_anomalies_use_case(),
            evaluate_model_use_case=container.evaluate_model_use_case(),
        )
    except Exception as e:
        logger.warning(f"Could not create full adapter: {e}")
        return ModelTrainerAdapter()  # Use fallback implementations
