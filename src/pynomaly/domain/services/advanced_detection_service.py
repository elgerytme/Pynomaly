"""Domain service for advanced anomaly detection with clean architecture."""

import logging
import time

import numpy as np

from ..entities.dataset import Dataset
from ..entities.detection_result import DetectionResult
from ..protocols.detection_protocols import (
    AdvancedDetectionService,
    AlgorithmFactoryProtocol,
    DetectionAlgorithm,
    DetectionConfig,
    DetectionMetrics,
    EnsembleAggregatorProtocol,
    EnsembleConfig,
    EnsembleMethod,
    PreprocessingMethod,
)

logger = logging.getLogger(__name__)


class DomainAdvancedDetectionService(AdvancedDetectionService):
    """Domain service for advanced anomaly detection using dependency injection."""

    def __init__(
        self,
        algorithm_factory: AlgorithmFactoryProtocol,
        ensemble_aggregator: EnsembleAggregatorProtocol
    ):
        """Initialize the service with injected dependencies."""
        super().__init__(algorithm_factory, ensemble_aggregator)
        self.logger = logger

    async def detect_anomalies_single(
        self,
        dataset: Dataset,
        config: DetectionConfig
    ) -> DetectionResult:
        """Detect anomalies using a single algorithm."""
        start_time = time.time()

        try:
            # Prepare data with preprocessing
            processed_data, preprocessor = self._prepare_data(dataset, config.preprocessing)

            # Create algorithm adapter
            algorithm = self.algorithm_factory.create_algorithm(config)

            # Fit and predict
            algorithm.fit(processed_data)
            predictions = algorithm.predict(processed_data)
            scores = algorithm.decision_function(processed_data)

            # Convert predictions to binary (1 for anomaly, 0 for normal)
            # Most algorithms return -1 for anomaly, 1 for normal
            if np.any(predictions == -1):
                predictions = np.where(predictions == -1, 1, 0)

            processing_time = time.time() - start_time

            # Create and return detection result
            return self._create_detection_result(
                dataset, predictions, scores, config, processing_time
            )

        except Exception as e:
            self.logger.error(f"Error in single algorithm detection: {e}")
            raise

    async def detect_anomalies_ensemble(
        self,
        dataset: Dataset,
        ensemble_config: EnsembleConfig
    ) -> DetectionResult:
        """Detect anomalies using ensemble methods."""
        start_time = time.time()

        try:
            all_predictions = []
            all_scores = []

            # Run each algorithm in the ensemble
            for config in ensemble_config.algorithms:
                # Prepare data with preprocessing
                processed_data, _ = self._prepare_data(dataset, config.preprocessing)

                # Create algorithm adapter
                algorithm = self.algorithm_factory.create_algorithm(config)

                # Fit and predict
                algorithm.fit(processed_data)
                predictions = algorithm.predict(processed_data)
                scores = algorithm.decision_function(processed_data)

                # Convert predictions to binary
                if np.any(predictions == -1):
                    predictions = np.where(predictions == -1, 1, 0)

                all_predictions.append(predictions)
                all_scores.append(scores)

            # Aggregate results using ensemble method
            ensemble_predictions = self.ensemble_aggregator.aggregate_predictions(
                all_predictions,
                ensemble_config.ensemble_method,
                ensemble_config.weights
            )

            ensemble_scores = self.ensemble_aggregator.aggregate_scores(
                all_scores,
                ensemble_config.ensemble_method,
                ensemble_config.weights
            )

            processing_time = time.time() - start_time

            # Create a synthetic config for the ensemble result
            ensemble_result_config = DetectionConfig(
                algorithm=DetectionAlgorithm.ISOLATION_FOREST,  # Placeholder
                contamination_rate=ensemble_config.algorithms[0].contamination_rate,
                preprocessing=PreprocessingMethod.NONE
            )

            return self._create_detection_result(
                dataset, ensemble_predictions, ensemble_scores,
                ensemble_result_config, processing_time
            )

        except Exception as e:
            self.logger.error(f"Error in ensemble detection: {e}")
            raise

    async def evaluate_algorithm_performance(
        self,
        dataset: Dataset,
        ground_truth: np.ndarray | None,
        config: DetectionConfig
    ) -> DetectionMetrics:
        """Evaluate algorithm performance against ground truth."""
        start_time = time.time()

        try:
            # Run detection
            result = await self.detect_anomalies_single(dataset, config)

            # Extract predictions
            predictions = np.zeros(len(dataset.data))
            for anomaly in result.anomalies:
                if 'index' in anomaly.context:
                    predictions[anomaly.context['index']] = 1

            processing_time = time.time() - start_time

            # Calculate basic metrics
            total_samples = len(dataset.data)
            total_anomalies = len(result.anomalies)
            anomaly_rate = total_anomalies / total_samples if total_samples > 0 else 0.0

            metrics = DetectionMetrics(
                total_samples=total_samples,
                total_anomalies=total_anomalies,
                anomaly_rate=anomaly_rate,
                processing_time_seconds=processing_time
            )

            # Add performance metrics if ground truth is available
            if ground_truth is not None:
                metrics = self._calculate_performance_metrics(
                    predictions, ground_truth, metrics
                )

            return metrics

        except Exception as e:
            self.logger.error(f"Error evaluating algorithm performance: {e}")
            raise

    async def compare_algorithms(
        self,
        dataset: Dataset,
        configs: list[DetectionConfig],
        ground_truth: np.ndarray | None = None
    ) -> dict[str, DetectionMetrics]:
        """Compare multiple algorithms on the same dataset."""
        results = {}

        for config in configs:
            try:
                metrics = await self.evaluate_algorithm_performance(
                    dataset, ground_truth, config
                )
                results[config.algorithm.value] = metrics

            except Exception as e:
                self.logger.error(f"Error comparing algorithm {config.algorithm.value}: {e}")
                continue

        return results

    async def auto_tune_parameters(
        self,
        dataset: Dataset,
        algorithm: DetectionAlgorithm,
        ground_truth: np.ndarray | None = None
    ) -> DetectionConfig:
        """Automatically tune algorithm parameters."""
        # This is a simplified implementation
        # In a real scenario, you'd use libraries like Optuna for hyperparameter optimization

        best_config = DetectionConfig(algorithm=algorithm)
        best_score = -float('inf')

        # Try different contamination rates
        contamination_rates = [0.05, 0.1, 0.15, 0.2]

        for contamination_rate in contamination_rates:
            try:
                config = DetectionConfig(
                    algorithm=algorithm,
                    contamination_rate=contamination_rate
                )

                metrics = await self.evaluate_algorithm_performance(
                    dataset, ground_truth, config
                )

                # Use anomaly rate as a simple scoring metric
                # In practice, you'd use more sophisticated metrics
                score = metrics.anomaly_rate

                if score > best_score:
                    best_score = score
                    best_config = config

            except Exception as e:
                self.logger.warning(f"Error tuning parameters for {algorithm.value}: {e}")
                continue

        return best_config

    def _calculate_performance_metrics(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        metrics: DetectionMetrics
    ) -> DetectionMetrics:
        """Calculate performance metrics against ground truth."""
        try:
            # Basic classification metrics
            true_positives = np.sum((predictions == 1) & (ground_truth == 1))
            false_positives = np.sum((predictions == 1) & (ground_truth == 0))
            false_negatives = np.sum((predictions == 0) & (ground_truth == 1))

            # Calculate precision, recall, F1
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            metrics.precision = precision
            metrics.recall = recall
            metrics.f1_score = f1_score

            # Note: AUC calculation would require scores, not just predictions
            # This would be implemented by the infrastructure layer

        except Exception as e:
            self.logger.warning(f"Error calculating performance metrics: {e}")

        return metrics

    def get_supported_algorithms(self) -> list[DetectionAlgorithm]:
        """Get list of supported algorithms from the factory."""
        try:
            return self.algorithm_factory.get_available_algorithms()
        except Exception as e:
            self.logger.error(f"Error getting supported algorithms: {e}")
            return []

    def create_default_ensemble_config(
        self,
        contamination_rate: float = 0.1
    ) -> EnsembleConfig:
        """Create a default ensemble configuration with multiple algorithms."""
        algorithms = [
            DetectionConfig(
                algorithm=DetectionAlgorithm.ISOLATION_FOREST,
                contamination_rate=contamination_rate
            ),
            DetectionConfig(
                algorithm=DetectionAlgorithm.ONE_CLASS_SVM,
                contamination_rate=contamination_rate
            ),
            DetectionConfig(
                algorithm=DetectionAlgorithm.LOCAL_OUTLIER_FACTOR,
                contamination_rate=contamination_rate
            )
        ]

        return EnsembleConfig(
            algorithms=algorithms,
            ensemble_method=EnsembleMethod.AVERAGE
        )

    def validate_dataset(self, dataset: Dataset) -> bool:
        """Validate that dataset is suitable for anomaly detection."""
        try:
            # Basic validation
            if dataset.data is None or len(dataset.data) == 0:
                return False

            # Check for minimum samples
            if len(dataset.data) < 10:
                self.logger.warning("Dataset has fewer than 10 samples, results may be unreliable")

            # Check for NaN values
            if np.any(np.isnan(dataset.data)):
                self.logger.warning("Dataset contains NaN values")
                return False

            # Check for infinite values
            if np.any(np.isinf(dataset.data)):
                self.logger.warning("Dataset contains infinite values")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating dataset: {e}")
            return False
