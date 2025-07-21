"""Detection pipeline service for autonomous detection."""

from __future__ import annotations

import logging
import time
from typing import Any

from .algorithm_adapter_registry import (
    AlgorithmAdapterRegistry,
)
from .algorithm_recommendation_service import (
    AlgorithmRecommendation,
)
from ...domain.entities import Dataset, DetectionResult, Detector
from ...domain.exceptions import AdapterError, AlgorithmNotFoundError
from ...shared.protocols import (
    DetectionResultRepositoryProtocol,
    DetectorRepositoryProtocol,
)


class DetectionPipelineService:
    """Service responsible for executing detection pipelines."""

    def __init__(
        self,
        detector_repository: DetectorRepositoryProtocol,
        result_repository: DetectionResultRepositoryProtocol,
        adapter_registry: AlgorithmAdapterRegistry | None = None,
    ):
        """Initialize detection pipeline service.

        Args:
            detector_repository: Repository for detectors
            result_repository: Repository for results
            adapter_registry: Registry of algorithm adapters
        """
        self.detector_repository = detector_repository
        self.result_repository = result_repository
        self.adapter_registry = adapter_registry or AlgorithmAdapterRegistry()
        self.logger = logging.getLogger(__name__)

    async def run_detection_pipeline(
        self,
        dataset: Dataset,
        recommendations: list[AlgorithmRecommendation],
        auto_tune: bool = True,
        save_results: bool = True,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Run detection pipeline with recommended algorithms.

        Args:
            dataset: Dataset to analyze
            recommendations: List of algorithm recommendations
            auto_tune: Whether to auto-tune hyperparameters
            save_results: Whether to save results to repository
            verbose: Enable verbose logging

        Returns:
            Dictionary containing detection results and metadata
        """
        if verbose:
            self.logger.info(f"Running detection pipeline with {len(recommendations)} algorithms")

        pipeline_results = {
            "dataset_name": dataset.name,
            "algorithms_used": [],
            "results": {},
            "performance_metrics": {},
            "execution_times": {},
            "best_algorithm": None,
            "best_score": 0.0,
            "ensemble_result": None,
            "errors": {},
        }

        # Run each recommended algorithm
        for i, recommendation in enumerate(recommendations):
            if verbose:
                self.logger.info(f"Running algorithm {i+1}/{len(recommendations)}: {recommendation.algorithm}")

            try:
                # Execute algorithm
                result = await self._execute_algorithm(
                    dataset=dataset,
                    recommendation=recommendation,
                    auto_tune=auto_tune,
                    verbose=verbose,
                )

                # Store results
                pipeline_results["algorithms_used"].append(recommendation.algorithm)
                pipeline_results["results"][recommendation.algorithm] = result
                pipeline_results["performance_metrics"][recommendation.algorithm] = self._calculate_performance_metrics(result)
                pipeline_results["execution_times"][recommendation.algorithm] = result.execution_time_ms

                # Save individual result if requested
                if save_results:
                    await self.result_repository.save(result)

                # Track best algorithm
                performance_score = self._calculate_overall_performance_score(result)
                if performance_score > pipeline_results["best_score"]:
                    pipeline_results["best_algorithm"] = recommendation.algorithm
                    pipeline_results["best_score"] = performance_score

            except Exception as e:
                self.logger.error(f"Error running {recommendation.algorithm}: {e}")
                pipeline_results["errors"][recommendation.algorithm] = str(e)

        # Create ensemble result if multiple algorithms succeeded
        successful_results = {
            alg: result for alg, result in pipeline_results["results"].items()
            if alg not in pipeline_results["errors"]
        }

        if len(successful_results) > 1:
            if verbose:
                self.logger.info("Creating ensemble result")

            try:
                ensemble_result = self._create_ensemble_result(dataset, successful_results, recommendations)
                pipeline_results["ensemble_result"] = ensemble_result

                if save_results:
                    await self.result_repository.save(ensemble_result)

                # Check if ensemble is better than best individual
                ensemble_score = self._calculate_overall_performance_score(ensemble_result)
                if ensemble_score > pipeline_results["best_score"]:
                    pipeline_results["best_algorithm"] = "ensemble"
                    pipeline_results["best_score"] = ensemble_score

            except Exception as e:
                self.logger.error(f"Error creating ensemble: {e}")
                pipeline_results["errors"]["ensemble"] = str(e)

        if verbose:
            self.logger.info(f"Pipeline completed. Best algorithm: {pipeline_results['best_algorithm']}")

        return pipeline_results

    async def _execute_algorithm(
        self,
        dataset: Dataset,
        recommendation: AlgorithmRecommendation,
        auto_tune: bool = True,
        verbose: bool = False,
    ) -> DetectionResult:
        """Execute a single algorithm.

        Args:
            dataset: Dataset to analyze
            recommendation: Algorithm recommendation
            auto_tune: Whether to auto-tune hyperparameters
            verbose: Enable verbose logging

        Returns:
            Detection result

        Raises:
            AdapterError: If algorithm execution fails
        """
        start_time = time.time()

        try:
            # Create detector
            detector = self._create_detector(recommendation)

            # Auto-tune if requested
            if auto_tune:
                detector = await self._auto_tune_detector(detector, dataset, verbose)

            # Fit and predict
            detection_result = detector.fit_detect(dataset)

            # Update execution time
            execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            detection_result.execution_time_ms = execution_time

            # Save detector
            await self.detector_repository.save(detector)

            return detection_result

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            raise AdapterError(f"Failed to execute {recommendation.algorithm}: {e}") from e

    def _create_detector(self, recommendation: AlgorithmRecommendation) -> Detector:
        """Create detector from recommendation.

        Args:
            recommendation: Algorithm recommendation

        Returns:
            Configured detector

        Raises:
            AlgorithmNotFoundError: If algorithm is not supported
        """
        try:
            # Get adapter class
            adapter_class = self.adapter_registry.get_adapter(recommendation.algorithm)

            # Create detector with recommended hyperparameters
            detector = adapter_class(
                algorithm_name=recommendation.algorithm,
                **recommendation.hyperparams
            )

            return detector

        except Exception as e:
            raise AlgorithmNotFoundError(f"Failed to create detector for {recommendation.algorithm}: {e}") from e

    async def _auto_tune_detector(
        self,
        detector: Detector,
        dataset: Dataset,
        verbose: bool = False,
    ) -> Detector:
        """Auto-tune detector hyperparameters.

        Args:
            detector: Detector to tune
            dataset: Dataset for tuning
            verbose: Enable verbose logging

        Returns:
            Tuned detector
        """
        if verbose:
            self.logger.info(f"Auto-tuning {detector.algorithm_name}")

        try:
            # Implement hyperparameter tuning based on detector type
            if hasattr(detector, 'get_params') and hasattr(detector, 'set_params'):
                # Get current parameters
                current_params = detector.get_params()

                # Define parameter spaces for different algorithms
                param_spaces = self._get_parameter_spaces(detector.algorithm_name)

                if param_spaces:
                    # Use grid search for hyperparameter tuning
                    best_params = await self._grid_search_tuning(
                        detector, dataset, param_spaces, verbose
                    )

                    # Update detector with best parameters
                    detector.set_params(**best_params)

                    if verbose:
                        self.logger.info(f"Tuned parameters for {detector.algorithm_name}: {best_params}")
                else:
                    if verbose:
                        self.logger.info(f"No parameter space defined for {detector.algorithm_name}")

            return detector

        except Exception as e:
            self.logger.warning(f"Auto-tuning failed for {detector.algorithm_name}: {e}")
            return detector

    def _get_parameter_spaces(self, algorithm_name: str) -> dict[str, list]:
        """Get parameter space for hyperparameter tuning."""
        param_spaces = {
            "IsolationForest": {
                "n_estimators": [50, 100, 200],
                "max_samples": [0.5, 0.8, 1.0],
                "contamination": [0.05, 0.1, 0.15, 0.2],
                "max_features": [0.5, 0.8, 1.0],
            },
            "LocalOutlierFactor": {
                "n_neighbors": [5, 10, 20, 35],
                "contamination": [0.05, 0.1, 0.15, 0.2],
                "algorithm": ["auto", "ball_tree", "kd_tree"],
                "leaf_size": [20, 30, 40],
            },
            "OneClassSVM": {
                "kernel": ["rbf", "linear", "poly"],
                "gamma": [0.001, 0.01, 0.1, 1.0],
                "nu": [0.05, 0.1, 0.15, 0.2],
            },
            "EllipticEnvelope": {
                "contamination": [0.05, 0.1, 0.15, 0.2],
                "support_fraction": [0.5, 0.7, 0.9],
            },
            "DBSCAN": {
                "eps": [0.1, 0.3, 0.5, 0.7],
                "min_samples": [3, 5, 10, 15],
                "algorithm": ["auto", "ball_tree", "kd_tree"],
            },
        }

        return param_spaces.get(algorithm_name, {})

    async def _grid_search_tuning(
        self,
        detector,
        dataset,
        param_spaces: dict[str, list],
        verbose: bool = False
    ) -> dict[str, Any]:
        """Perform grid search hyperparameter tuning."""
        import itertools


        best_params = {}
        best_score = float('-inf')

        # Generate all parameter combinations
        param_names = list(param_spaces.keys())
        param_values = list(param_spaces.values())

        # Limit combinations to avoid excessive computation
        max_combinations = 20
        all_combinations = list(itertools.product(*param_values))

        if len(all_combinations) > max_combinations:
            # Sample random combinations if too many
            import random
            random.seed(42)
            combinations = random.sample(all_combinations, max_combinations)
        else:
            combinations = all_combinations

        if verbose:
            self.logger.info(f"Testing {len(combinations)} parameter combinations")

        for i, param_values in enumerate(combinations):
            try:
                # Create parameter dictionary
                params = dict(zip(param_names, param_values, strict=False))

                # Set parameters
                detector.set_params(**params)

                # Evaluate detector performance
                score = await self._evaluate_detector_performance(detector, dataset)

                if verbose and i % 5 == 0:
                    self.logger.info(f"Combination {i+1}/{len(combinations)}: {params} -> Score: {score:.4f}")

                # Update best parameters if better
                if score > best_score:
                    best_score = score
                    best_params = params.copy()

            except Exception as e:
                if verbose:
                    self.logger.warning(f"Parameter combination failed: {params} -> {e}")
                continue

        return best_params

    async def _evaluate_detector_performance(self, detector, dataset) -> float:
        """Evaluate detector performance using cross-validation or simple metrics."""
        try:
            # Fit detector
            detector.fit(dataset)

            # Get anomaly scores
            scores = detector.score(dataset)

            # Calculate performance metric (higher is better)
            # For unsupervised learning, we use silhouette score or similar
            if hasattr(dataset, 'labels') and dataset.labels is not None:
                # If we have labels, use AUC
                labels = dataset.labels
                score_values = [s.value for s in scores]

                try:
                    from sklearn.metrics import roc_auc_score
                    return roc_auc_score(labels, score_values)
                except:
                    # Fallback to simple score variance
                    import numpy as np
                    return float(np.var(score_values))
            else:
                # No labels - use score variance as proxy for separation
                import numpy as np
                score_values = [s.value for s in scores]
                return float(np.var(score_values))

        except Exception:
            # Return poor score if evaluation fails
            return 0.0

    def _calculate_performance_metrics(self, result: DetectionResult) -> dict[str, float]:
        """Calculate performance metrics for a detection result.

        Args:
            result: Detection result

        Returns:
            Dictionary of performance metrics
        """
        try:
            metrics = {}

            # Basic metrics
            total_samples = len(result.labels)
            anomaly_count = sum(result.labels)
            normal_count = total_samples - anomaly_count

            metrics["total_samples"] = total_samples
            metrics["anomaly_count"] = anomaly_count
            metrics["normal_count"] = normal_count
            metrics["anomaly_rate"] = anomaly_count / total_samples if total_samples > 0 else 0

            # Score statistics
            if result.scores:
                score_values = [score.value for score in result.scores]
                metrics["mean_score"] = sum(score_values) / len(score_values)
                metrics["min_score"] = min(score_values)
                metrics["max_score"] = max(score_values)
                metrics["score_std"] = self._calculate_std(score_values)

            # Threshold metrics
            metrics["threshold"] = result.threshold
            metrics["execution_time_ms"] = result.execution_time_ms

            return metrics

        except Exception as e:
            self.logger.warning(f"Failed to calculate performance metrics: {e}")
            return {}

    def _calculate_std(self, values: list[float]) -> float:
        """Calculate standard deviation of values.

        Args:
            values: List of values

        Returns:
            Standard deviation
        """
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5

    def _calculate_overall_performance_score(self, result: DetectionResult) -> float:
        """Calculate overall performance score for a detection result.

        Args:
            result: Detection result

        Returns:
            Overall performance score between 0 and 1
        """
        try:
            score = 0.0
            weight_sum = 0.0

            # Score quality (variance and distribution)
            if result.scores:
                score_values = [score.value for score in result.scores]
                score_std = self._calculate_std(score_values)
                score_range = max(score_values) - min(score_values)

                # Good scores have reasonable variance and range
                if score_range > 0:
                    score_quality = min(1.0, score_std / score_range)
                    score += score_quality * 0.3
                    weight_sum += 0.3

            # Anomaly rate reasonableness (not too high or too low)
            anomaly_rate = sum(result.labels) / len(result.labels) if result.labels else 0
            if 0.01 <= anomaly_rate <= 0.5:  # Reasonable range
                rate_quality = 1.0 - abs(anomaly_rate - 0.1)  # Prefer ~10% anomalies
                score += rate_quality * 0.2
                weight_sum += 0.2

            # Execution time (faster is better, but not the most important)
            if result.execution_time_ms > 0:
                # Normalize execution time (prefer under 10 seconds)
                time_quality = max(0.0, 1.0 - result.execution_time_ms / 10000)
                score += time_quality * 0.1
                weight_sum += 0.1

            # Threshold quality (should be reasonable)
            if 0.1 <= result.threshold <= 0.9:
                threshold_quality = 1.0 - abs(result.threshold - 0.5)
                score += threshold_quality * 0.1
                weight_sum += 0.1

            # Metadata completeness
            if result.metadata:
                metadata_quality = min(1.0, len(result.metadata) / 5)  # Prefer rich metadata
                score += metadata_quality * 0.1
                weight_sum += 0.1

            # Base score for successful execution
            score += 0.2
            weight_sum += 0.2

            return score / weight_sum if weight_sum > 0 else 0.0

        except Exception as e:
            self.logger.warning(f"Failed to calculate performance score: {e}")
            return 0.0

    def _create_ensemble_result(
        self,
        dataset: Dataset,
        results: dict[str, DetectionResult],
        recommendations: list[AlgorithmRecommendation],
    ) -> DetectionResult:
        """Create ensemble result from multiple detection results.

        Args:
            dataset: Original dataset
            results: Dictionary of detection results by algorithm
            recommendations: List of algorithm recommendations

        Returns:
            Ensemble detection result
        """
        try:
            # Create ensemble scores by averaging
            ensemble_scores = []
            ensemble_labels = []

            # Get all score values
            algorithm_scores = {}
            for alg_name, result in results.items():
                if result.scores:
                    algorithm_scores[alg_name] = [score.value for score in result.scores]

            # Calculate ensemble scores
            if algorithm_scores:
                n_samples = len(list(algorithm_scores.values())[0])

                for i in range(n_samples):
                    # Average scores across algorithms
                    sample_scores = []
                    for alg_name, scores in algorithm_scores.items():
                        if i < len(scores):
                            sample_scores.append(scores[i])

                    if sample_scores:
                        ensemble_score = sum(sample_scores) / len(sample_scores)
                        ensemble_scores.append(ensemble_score)

                # Calculate ensemble threshold
                ensemble_threshold = sum(result.threshold for result in results.values()) / len(results)

                # Create ensemble labels
                ensemble_labels = [1 if score > ensemble_threshold else 0 for score in ensemble_scores]

            # Create ensemble metadata
            ensemble_metadata = {
                "ensemble_type": "simple_average",
                "algorithms": list(results.keys()),
                "algorithm_weights": {alg: 1.0 / len(results) for alg in results.keys()},
                "individual_thresholds": {alg: result.threshold for alg, result in results.items()},
                "ensemble_threshold": ensemble_threshold,
                "total_execution_time_ms": sum(result.execution_time_ms for result in results.values()),
            }

            # Create ensemble result
            from uuid import uuid4

            from ...domain.value_objects import AnomalyScore

            ensemble_result = DetectionResult(
                detector_id=str(uuid4()),
                dataset_name=dataset.name,
                scores=[AnomalyScore(value=score, threshold=ensemble_threshold) for score in ensemble_scores],
                labels=ensemble_labels,
                threshold=ensemble_threshold,
                execution_time_ms=sum(result.execution_time_ms for result in results.values()),
                anomalies=[],  # Could be populated from high-scoring samples
                metadata=ensemble_metadata,
            )

            return ensemble_result

        except Exception as e:
            self.logger.error(f"Failed to create ensemble result: {e}")
            raise AdapterError(f"Ensemble creation failed: {e}") from e

    def get_pipeline_summary(self, pipeline_results: dict[str, Any]) -> str:
        """Generate human-readable summary of pipeline results.

        Args:
            pipeline_results: Pipeline results dictionary

        Returns:
            Human-readable summary
        """
        try:
            summary_lines = [
                "=== DETECTION PIPELINE SUMMARY ===",
                f"Dataset: {pipeline_results['dataset_name']}",
                f"Algorithms Executed: {len(pipeline_results['algorithms_used'])}",
                f"Successful Algorithms: {len(pipeline_results['results'])}",
                f"Best Algorithm: {pipeline_results['best_algorithm']}",
                f"Best Score: {pipeline_results['best_score']:.3f}",
            ]

            if pipeline_results["errors"]:
                summary_lines.append(f"Errors: {len(pipeline_results['errors'])}")

            # Individual algorithm results
            summary_lines.append("\n=== INDIVIDUAL RESULTS ===")
            for alg in pipeline_results["algorithms_used"]:
                if alg in pipeline_results["results"]:
                    result = pipeline_results["results"][alg]
                    metrics = pipeline_results["performance_metrics"][alg]
                    summary_lines.append(
                        f"{alg}: {metrics['anomaly_count']} anomalies, "
                        f"{result.execution_time_ms:.0f}ms"
                    )
                elif alg in pipeline_results["errors"]:
                    summary_lines.append(f"{alg}: ERROR - {pipeline_results['errors'][alg]}")

            # Ensemble result
            if pipeline_results["ensemble_result"]:
                result = pipeline_results["ensemble_result"]
                summary_lines.append(f"\nEnsemble: {sum(result.labels)} anomalies, "
                                    f"{result.execution_time_ms:.0f}ms")

            return "\n".join(summary_lines)

        except Exception as e:
            self.logger.warning(f"Failed to generate pipeline summary: {e}")
            return "Failed to generate pipeline summary"