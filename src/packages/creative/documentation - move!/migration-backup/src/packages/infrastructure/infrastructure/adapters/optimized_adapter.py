"""Optimized adapter that integrates algorithm optimization capabilities.

This adapter wraps existing adapters with optimization functionality,
providing enhanced performance through intelligent parameter tuning.
"""

from __future__ import annotations

from typing import Any

from ...application.services.algorithm_optimization_service import (
    AlgorithmOptimizationService,
)
from ...domain.entities import Dataset, DetectionResult, Detector
from ...infrastructure.config.feature_flags import require_feature
from .pyod_adapter import PyODAdapter
from .sklearn_adapter import SklearnAdapter


class OptimizedAdapter:
    """Adapter that provides optimized anomaly detection."""

    def __init__(
        self,
        detector: Detector,
        base_adapter_class: type | None = None,
        optimization_level: str = "balanced",
        auto_optimize: bool = True,
    ):
        """Initialize optimized adapter.

        Args:
            detector: The detector to optimize
            base_adapter_class: Base adapter class to wrap
            optimization_level: Level of optimization ("fast", "balanced", "thorough")
            auto_optimize: Whether to automatically optimize on first fit
        """
        self.detector = detector

        self.optimization_level = optimization_level
        self.auto_optimize = auto_optimize
        self.optimization_service = AlgorithmOptimizationService()

        # Determine base adapter if not provided
        if base_adapter_class is None:
            base_adapter_class = self._determine_base_adapter_class()

        self.base_adapter_class = base_adapter_class
        self.base_adapter: Any | None = None
        self.optimized_detector: Detector | None = None
        self.optimization_results: dict[str, Any] | None = None
        self.is_optimized = False

    @require_feature("algorithm_optimization")
    def fit(self, dataset: Dataset) -> None:
        """Fit the detector with optimization."""
        # Perform optimization if enabled and not already done
        if self.auto_optimize and not self.is_optimized:
            self._optimize_for_dataset(dataset)

        # Create base adapter with optimized detector
        detector_to_use = (
            self.optimized_detector if self.is_optimized else self.detector
        )
        self.base_adapter = self.base_adapter_class(detector_to_use)

        # Fit the base adapter
        self.base_adapter.fit(dataset)

    def detect(self, dataset: Dataset) -> DetectionResult:
        """Perform anomaly detection."""
        if self.base_adapter is None:
            raise RuntimeError("Adapter must be fitted before detection")

        return self.base_adapter.detect(dataset)

    def score(self, dataset: Dataset) -> float:
        """Get anomaly scores."""
        if self.base_adapter is None:
            raise RuntimeError("Adapter must be fitted before scoring")

        return self.base_adapter.score(dataset)

    @require_feature("algorithm_optimization")
    def optimize_for_dataset(
        self,
        dataset: Dataset,
        optimization_level: str | None = None,
        contamination_rate: float | None = None,
    ) -> dict[str, Any]:
        """Explicitly optimize the detector for a specific dataset.

        Args:
            dataset: Dataset to optimize for
            optimization_level: Override default optimization level
            contamination_rate: Expected contamination rate

        Returns:
            Optimization results
        """
        level = optimization_level or self.optimization_level

        (
            self.optimized_detector,
            self.optimization_results,
        ) = self.optimization_service.optimize_detector(
            self.detector, dataset, level, contamination_rate=contamination_rate
        )

        self.is_optimized = True

        return self.optimization_results

    @require_feature("algorithm_optimization")
    def get_adaptive_parameters(
        self, dataset: Dataset, performance_targets: dict[str, float] | None = None
    ) -> dict[str, Any]:
        """Get adaptive parameters for the dataset.

        Args:
            dataset: Dataset for parameter adaptation
            performance_targets: Target performance metrics

        Returns:
            Adaptive parameter configuration
        """
        return self.optimization_service.adaptive_parameter_selection(
            self.detector.algorithm_name, dataset, performance_targets
        )

    @require_feature("algorithm_optimization")
    def benchmark_optimization_impact(
        self, dataset: Dataset, n_iterations: int = 3
    ) -> dict[str, Any]:
        """Benchmark the impact of optimization.

        Args:
            dataset: Test dataset
            n_iterations: Number of benchmark iterations

        Returns:
            Benchmark results
        """
        return self.optimization_service.benchmark_optimization_impact(
            self.detector, dataset, n_iterations
        )

    def get_optimization_status(self) -> dict[str, Any]:
        """Get current optimization status."""
        return {
            "is_optimized": self.is_optimized,
            "optimization_level": self.optimization_level,
            "auto_optimize": self.auto_optimize,
            "base_adapter_class": (
                self.base_adapter_class.__name__ if self.base_adapter_class else None
            ),
            "optimization_results": self.optimization_results,
            "optimized_parameters": (
                self.optimized_detector.parameters if self.optimized_detector else None
            ),
        }

    def reset_optimization(self) -> None:
        """Reset optimization state."""
        self.optimized_detector = None
        self.optimization_results = None
        self.is_optimized = False
        self.base_adapter = None

    def _optimize_for_dataset(self, dataset: Dataset) -> None:
        """Internal method to optimize for dataset."""
        try:
            (
                self.optimized_detector,
                self.optimization_results,
            ) = self.optimization_service.optimize_detector(
                self.detector, dataset, self.optimization_level
            )
            self.is_optimized = True
        except Exception as e:
            # If optimization fails, continue with original detector
            import warnings

            warnings.warn(
                f"Optimization failed, using original detector: {e}", stacklevel=2
            )
            self.is_optimized = False

    def _determine_base_adapter_class(self) -> type:
        """Determine appropriate base adapter class."""
        algorithm_name = self.detector.algorithm_name.lower()

        # PyOD algorithms
        pyod_algorithms = {
            "knn",
            "abod",
            "hbos",
            "pca",
            "mcd",
            "ocsvm",
            "lmdd",
            "cblof",
            "feature_bagging",
            "iforest",
            "histogram",
        }

        # Sklearn algorithms
        sklearn_algorithms = {
            "isolationforest",
            "localoutlierfactor",
            "oneclasssvm",
            "ellipticenvelope",
            "svm",
        }

        if algorithm_name in pyod_algorithms:
            return PyODAdapter
        elif algorithm_name in sklearn_algorithms:
            return SklearnAdapter
        else:
            # Default to sklearn for unknown algorithms
            return SklearnAdapter


class OptimizedEnsembleAdapter:
    """Adapter for optimized ensemble detection."""

    def __init__(
        self,
        detectors: list[Detector],
        ensemble_strategy: str = "voting",
        optimization_level: str = "balanced",
    ):
        """Initialize optimized ensemble adapter.

        Args:
            detectors: List of detectors for ensemble
            ensemble_strategy: Ensemble combination strategy
            optimization_level: Optimization level for individual detectors
        """
        # Store detectors
        self.detectors = detectors
        self.ensemble_strategy = ensemble_strategy
        self.optimization_level = optimization_level
        self.optimization_service = AlgorithmOptimizationService()

        self.optimized_adapters: list[OptimizedAdapter] = []
        self.ensemble_weights: list[float] | None = None
        self.ensemble_results: dict[str, Any] | None = None
        self.is_fitted = False

    @require_feature("algorithm_optimization")
    def fit(self, dataset: Dataset) -> None:
        """Fit the ensemble with optimization."""
        # Optimize entire ensemble
        (
            optimized_detectors,
            self.ensemble_results,
        ) = self.optimization_service.optimize_ensemble(
            self.detectors, dataset, self.ensemble_strategy, self.optimization_level
        )

        # Create optimized adapters
        self.optimized_adapters = []
        for detector in optimized_detectors:
            adapter = OptimizedAdapter(
                detector,
                optimization_level=self.optimization_level,
                auto_optimize=False,  # Already optimized
            )
            adapter.optimized_detector = detector
            adapter.is_optimized = True
            self.optimized_adapters.append(adapter)

        # Fit all adapters
        for adapter in self.optimized_adapters:
            adapter.fit(dataset)

        # Extract ensemble weights
        self.ensemble_weights = self.ensemble_results.get(
            "combination_weights",
            [1.0 / len(self.optimized_adapters)] * len(self.optimized_adapters),
        )

        self.is_fitted = True

    def detect(self, dataset: Dataset) -> DetectionResult:
        """Perform ensemble anomaly detection."""
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before detection")

        # Get results from all adapters
        results = []
        for adapter in self.optimized_adapters:
            result = adapter.detect(dataset)
            results.append(result)

        # Combine results based on strategy
        if self.ensemble_strategy == "voting":
            return self._combine_by_voting(results)
        elif self.ensemble_strategy == "weighted":
            return self._combine_by_weights(results)
        elif self.ensemble_strategy == "stacking":
            return self._combine_by_stacking(results)
        else:
            return self._combine_by_voting(results)  # Default

    def score(self, dataset: Dataset) -> float:
        """Get ensemble anomaly scores."""
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before scoring")

        scores = []
        for adapter in self.optimized_adapters:
            score = adapter.score(dataset)
            scores.append(score)

        # Weighted average
        weighted_score = sum(
            score * weight
            for score, weight in zip(scores, self.ensemble_weights, strict=False)
        )

        return weighted_score

    def get_ensemble_status(self) -> dict[str, Any]:
        """Get ensemble optimization status."""
        return {
            "n_detectors": len(self.detectors),
            "ensemble_strategy": self.ensemble_strategy,
            "optimization_level": self.optimization_level,
            "is_fitted": self.is_fitted,
            "ensemble_weights": self.ensemble_weights,
            "ensemble_results": self.ensemble_results,
            "individual_optimizations": [
                adapter.get_optimization_status() for adapter in self.optimized_adapters
            ],
        }

    def _combine_by_voting(self, results: list[DetectionResult]) -> DetectionResult:
        """Combine results by majority voting."""
        import numpy as np

        # Combine labels by majority vote
        all_labels = np.array([result.labels for result in results])
        combined_labels = np.round(np.mean(all_labels, axis=0)).astype(int)

        # Combine scores by average
        all_scores = np.array([result.scores for result in results])
        combined_scores = np.mean(all_scores, axis=0)

        # Use average threshold
        avg_threshold = np.mean([result.threshold for result in results])

        return DetectionResult(
            scores=combined_scores,
            labels=combined_labels,
            threshold=avg_threshold,
            metadata={
                "ensemble_strategy": "voting",
                "n_detectors": len(results),
                "individual_thresholds": [r.threshold for r in results],
            },
        )

    def _combine_by_weights(self, results: list[DetectionResult]) -> DetectionResult:
        """Combine results by weighted average."""
        import numpy as np

        # Weighted combination of scores
        weighted_scores = np.zeros_like(results[0].scores)
        for result, weight in zip(results, self.ensemble_weights, strict=False):
            weighted_scores += result.scores * weight

        # Weighted combination of labels
        weighted_labels = np.zeros_like(results[0].labels, dtype=float)
        for result, weight in zip(results, self.ensemble_weights, strict=False):
            weighted_labels += result.labels.astype(float) * weight

        combined_labels = np.round(weighted_labels).astype(int)

        # Weighted threshold
        weighted_threshold = sum(
            result.threshold * weight
            for result, weight in zip(results, self.ensemble_weights, strict=False)
        )

        return DetectionResult(
            scores=weighted_scores,
            labels=combined_labels,
            threshold=weighted_threshold,
            metadata={
                "ensemble_strategy": "weighted",
                "weights": self.ensemble_weights,
                "n_detectors": len(results),
            },
        )

    def _combine_by_stacking(self, results: list[DetectionResult]) -> DetectionResult:
        """Combine results by stacking (simplified)."""
        # For simplicity, use weighted combination for now
        # In a full implementation, this would train a meta-classifier
        return self._combine_by_weights(results)
