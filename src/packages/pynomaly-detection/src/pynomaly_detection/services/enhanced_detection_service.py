"""Enhanced detection service integrating with the new algorithm system."""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pynomaly.domain.entities import Dataset, DetectionResult
from pynomaly.domain.exceptions import DetectorNotFittedError, FittingError
from pynomaly.domain.value_objects import ContaminationRate
from pynomaly.infrastructure.adapters.algorithm_factory import (
    AlgorithmFactory,
    AlgorithmRecommendation,
    DatasetCharacteristics,
)
from pynomaly.infrastructure.adapters.ensemble_meta_adapter import (
    AggregationMethod,
    EnsembleMetaAdapter,
)
from pynomaly.shared.protocols import DetectorProtocol


class EnhancedDetectionService:
    """Enhanced detection service with advanced algorithm management."""

    def __init__(
        self,
        algorithm_factory: AlgorithmFactory | None = None,
        max_workers: int = 4,
        enable_caching: bool = True,
    ):
        """Initialize enhanced detection service.

        Args:
            algorithm_factory: Factory for creating algorithms
            max_workers: Maximum number of parallel workers
            enable_caching: Whether to enable result caching
        """
        self.algorithm_factory = algorithm_factory or AlgorithmFactory()
        self.max_workers = max_workers
        self.enable_caching = enable_caching

        # Cache for fitted detectors and results
        self._detector_cache: dict[str, DetectorProtocol] = {}
        self._result_cache: dict[str, DetectionResult] = {}

        # Performance tracking
        self._performance_history: list[dict[str, Any]] = []

    async def auto_detect(
        self,
        dataset: Dataset,
        performance_preference: str = "balanced",
        contamination_rate: ContaminationRate | None = None,
        save_detector: bool = True,
        detector_name: str | None = None,
    ) -> DetectionResult:
        """Automatically select and run the best detector for the dataset.

        Args:
            dataset: Dataset to analyze
            performance_preference: "fast", "balanced", or "accurate"
            contamination_rate: Expected contamination rate
            save_detector: Whether to cache the fitted detector
            detector_name: Custom name for the detector

        Returns:
            Detection result from the best algorithm
        """
        start_time = time.perf_counter()

        # Analyze dataset characteristics
        characteristics = self._analyze_dataset(dataset, contamination_rate)

        # Create auto-detector
        detector = self.algorithm_factory.create_auto_detector(
            dataset_characteristics=characteristics,
            performance_preference=performance_preference,
            name=detector_name,
            contamination_rate=contamination_rate,
        )

        # Fit and detect
        result = await self._fit_and_detect(detector, dataset)

        # Cache detector if requested
        if save_detector and detector_name:
            self._detector_cache[detector_name] = detector

        # Track performance
        total_time = time.perf_counter() - start_time
        self._track_performance(
            {
                "operation": "auto_detect",
                "dataset_name": dataset.name,
                "dataset_size": len(dataset.data),
                "algorithm": detector.name,
                "performance_preference": performance_preference,
                "total_time": total_time,
                "n_anomalies": len(result.anomalies),
            }
        )

        return result

    async def compare_algorithms(
        self,
        dataset: Dataset,
        algorithm_configs: list[dict[str, Any]] | None = None,
        contamination_rate: ContaminationRate | None = None,
        include_ensembles: bool = True,
        max_algorithms: int = 5,
    ) -> dict[str, Any]:
        """Compare multiple algorithms on the same dataset.

        Args:
            dataset: Dataset to analyze
            algorithm_configs: Specific algorithm configurations to test
            contamination_rate: Expected contamination rate
            include_ensembles: Whether to include ensemble methods
            max_algorithms: Maximum number of algorithms to compare

        Returns:
            Comprehensive comparison results
        """
        start_time = time.perf_counter()

        # Get algorithm recommendations if not provided
        if algorithm_configs is None:
            characteristics = self._analyze_dataset(dataset, contamination_rate)
            recommendations = self.algorithm_factory.recommend_algorithms(
                dataset_characteristics=characteristics,
                top_k=max_algorithms,
                include_ensembles=include_ensembles,
            )

            algorithm_configs = [
                {
                    "algorithm_name": rec.algorithm_name,
                    "library": rec.library,
                    "contamination_rate": contamination_rate,
                }
                for rec in recommendations
            ]

        # Create detectors
        detectors = []
        detector_names = []

        for config in algorithm_configs[:max_algorithms]:
            try:
                detector = self.algorithm_factory.create_detector(**config)
                detectors.append(detector)
                detector_names.append(detector.name)
            except Exception as e:
                print(f"Failed to create detector with config {config}: {e}")
                continue

        if not detectors:
            raise ValueError("No valid detectors could be created")

        # Run comparisons in parallel
        results = await self._run_parallel_detection(detectors, dataset)

        # Analyze results
        comparison = self._analyze_comparison_results(detector_names, results, dataset)

        # Add timing information
        comparison["metadata"] = {
            "total_comparison_time": time.perf_counter() - start_time,
            "n_algorithms": len(detectors),
            "dataset_size": len(dataset.data),
            "dataset_name": dataset.name,
        }

        return comparison

    async def create_ensemble_detector(
        self,
        dataset: Dataset,
        base_algorithms: list[str] | None = None,
        aggregation_method: (
            AggregationMethod | str
        ) = AggregationMethod.WEIGHTED_AVERAGE,
        contamination_rate: ContaminationRate | None = None,
        ensemble_name: str = "CustomEnsemble",
        auto_weight: bool = True,
    ) -> EnsembleMetaAdapter:
        """Create and fit an ensemble detector.

        Args:
            dataset: Dataset to train on
            base_algorithms: List of base algorithm names
            aggregation_method: Method for combining predictions
            contamination_rate: Expected contamination rate
            ensemble_name: Name for the ensemble
            auto_weight: Whether to automatically determine weights

        Returns:
            Fitted ensemble detector
        """
        # Get default algorithms if not provided
        if base_algorithms is None:
            characteristics = self._analyze_dataset(dataset, contamination_rate)
            recommendations = self.algorithm_factory.recommend_algorithms(
                characteristics, top_k=3, include_ensembles=False
            )
            base_algorithms = [rec.algorithm_name for rec in recommendations]

        # Create detector configurations
        detector_configs = []
        for algo in base_algorithms:
            try:
                # Detect library for the algorithm
                library = self.algorithm_factory._detect_library(algo)
                config = {
                    "algorithm_name": algo,
                    "library": library,
                    "contamination_rate": contamination_rate,
                    "weight": 1.0,  # Will be adjusted if auto_weight is True
                }
                detector_configs.append(config)
            except Exception as e:
                print(f"Skipping algorithm {algo}: {e}")
                continue

        if not detector_configs:
            raise ValueError("No valid algorithms provided for ensemble")

        # Create ensemble
        if isinstance(aggregation_method, str):
            aggregation_method = AggregationMethod(aggregation_method.lower())

        ensemble = self.algorithm_factory.create_ensemble(
            detector_configs=detector_configs,
            name=ensemble_name,
            contamination_rate=contamination_rate,
            aggregation_method=aggregation_method,
        )

        # Fit ensemble
        await self._fit_detector(ensemble, dataset)

        # Auto-weight if requested
        if auto_weight and len(ensemble.base_detectors) > 1:
            await self._optimize_ensemble_weights(ensemble, dataset)

        return ensemble

    async def recommend_and_create(
        self,
        dataset: Dataset,
        contamination_rate: ContaminationRate | None = None,
        top_k: int = 3,
        return_recommendations: bool = False,
    ) -> (
        list[DetectorProtocol]
        | tuple[list[DetectorProtocol], list[AlgorithmRecommendation]]
    ):
        """Get algorithm recommendations and create corresponding detectors.

        Args:
            dataset: Dataset for analysis
            contamination_rate: Expected contamination rate
            top_k: Number of recommendations
            return_recommendations: Whether to return recommendation details

        Returns:
            List of created detectors, optionally with recommendations
        """
        # Analyze dataset
        characteristics = self._analyze_dataset(dataset, contamination_rate)

        # Get recommendations
        recommendations = self.algorithm_factory.recommend_algorithms(
            dataset_characteristics=characteristics, top_k=top_k, include_ensembles=True
        )

        # Create detectors
        detectors = []
        for rec in recommendations:
            try:
                detector = self.algorithm_factory.create_detector(
                    algorithm_name=rec.algorithm_name,
                    library=rec.library,
                    contamination_rate=contamination_rate,
                )
                detectors.append(detector)
            except Exception as e:
                print(f"Failed to create detector for {rec.algorithm_name}: {e}")
                continue

        if return_recommendations:
            return detectors, recommendations
        return detectors

    async def batch_detect(
        self,
        detectors: list[DetectorProtocol],
        datasets: list[Dataset],
        fit_if_needed: bool = True,
    ) -> dict[str, dict[str, DetectionResult]]:
        """Run batch detection across multiple detectors and datasets.

        Args:
            detectors: List of detectors to use
            datasets: List of datasets to analyze
            fit_if_needed: Whether to fit detectors if not already fitted

        Returns:
            Nested dictionary: {detector_name: {dataset_name: result}}
        """
        results = {}

        # Process each detector
        for detector in detectors:
            detector_results = {}

            for dataset in datasets:
                try:
                    # Fit if needed
                    if not detector.is_fitted and fit_if_needed:
                        await self._fit_detector(detector, dataset)

                    # Detect
                    result = await self._detect_with_detector(detector, dataset)
                    detector_results[dataset.name] = result

                except Exception as e:
                    print(
                        f"Error with detector {detector.name} on dataset {dataset.name}: {e}"
                    )
                    continue

            results[detector.name] = detector_results

        return results

    def get_cached_detector(self, name: str) -> DetectorProtocol | None:
        """Get a cached detector by name."""
        return self._detector_cache.get(name)

    def cache_detector(self, name: str, detector: DetectorProtocol) -> None:
        """Cache a detector for later use."""
        self._detector_cache[name] = detector

    def clear_cache(self) -> None:
        """Clear all cached detectors and results."""
        self._detector_cache.clear()
        self._result_cache.clear()

    def get_performance_history(self) -> list[dict[str, Any]]:
        """Get performance tracking history."""
        return self._performance_history.copy()

    def _analyze_dataset(
        self, dataset: Dataset, contamination_rate: ContaminationRate | None = None
    ) -> DatasetCharacteristics:
        """Analyze dataset to extract characteristics."""
        # Basic characteristics
        n_samples = len(dataset.data)
        n_features = len(dataset.data.columns)

        # Check for categorical features
        has_categorical = any(
            dataset.data[col].dtype == "object"
            or dataset.data[col].dtype.name == "category"
            for col in dataset.data.columns
        )

        # Check for missing values
        has_missing_values = dataset.data.isnull().any().any()

        # Estimate contamination if not provided
        contamination_estimate = None
        if contamination_rate is not None:
            contamination_estimate = contamination_rate.value

        # Determine computational budget based on dataset size
        if n_samples < 1000:
            computational_budget = "high"
        elif n_samples < 10000:
            computational_budget = "medium"
        else:
            computational_budget = "low"

        return DatasetCharacteristics(
            n_samples=n_samples,
            n_features=n_features,
            has_categorical=has_categorical,
            has_missing_values=has_missing_values,
            contamination_estimate=contamination_estimate,
            computational_budget=computational_budget,
        )

    async def _fit_and_detect(
        self, detector: DetectorProtocol, dataset: Dataset
    ) -> DetectionResult:
        """Fit detector and run detection."""
        # Check cache first
        cache_key = f"{detector.name}_{hash(str(dataset.data.values.tobytes()))}"

        if self.enable_caching and cache_key in self._result_cache:
            return self._result_cache[cache_key]

        # Fit if needed
        if not detector.is_fitted:
            await self._fit_detector(detector, dataset)

        # Detect
        result = await self._detect_with_detector(detector, dataset)

        # Cache result
        if self.enable_caching:
            self._result_cache[cache_key] = result

        return result

    async def _fit_detector(self, detector: DetectorProtocol, dataset: Dataset) -> None:
        """Fit a detector asynchronously."""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            try:
                await loop.run_in_executor(executor, detector.fit, dataset)
            except Exception as e:
                raise FittingError(
                    detector_name=detector.name,
                    reason=str(e),
                    dataset_name=dataset.name,
                ) from e

    async def _detect_with_detector(
        self, detector: DetectorProtocol, dataset: Dataset
    ) -> DetectionResult:
        """Run detection with a detector asynchronously."""
        if not detector.is_fitted:
            raise DetectorNotFittedError(
                detector_name=detector.name, operation="detect"
            )

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            return await loop.run_in_executor(executor, detector.detect, dataset)

    async def _run_parallel_detection(
        self, detectors: list[DetectorProtocol], dataset: Dataset
    ) -> list[DetectionResult]:
        """Run detection with multiple detectors in parallel."""
        # Fit all detectors first
        fit_tasks = [
            self._fit_detector(detector, dataset)
            for detector in detectors
            if not detector.is_fitted
        ]

        if fit_tasks:
            await asyncio.gather(*fit_tasks, return_exceptions=True)

        # Run detection in parallel
        detect_tasks = [
            self._detect_with_detector(detector, dataset) for detector in detectors
        ]

        results = await asyncio.gather(*detect_tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = [
            result for result in results if isinstance(result, DetectionResult)
        ]

        return valid_results

    def _analyze_comparison_results(
        self,
        detector_names: list[str],
        results: list[DetectionResult],
        dataset: Dataset,
    ) -> dict[str, Any]:
        """Analyze and compare detection results."""
        comparison = {
            "individual_results": {},
            "summary": {},
            "rankings": {},
        }

        # Analyze individual results
        for name, result in zip(detector_names, results, strict=False):
            if isinstance(result, DetectionResult):
                comparison["individual_results"][name] = {
                    "n_anomalies": len(result.anomalies),
                    "anomaly_rate": len(result.anomalies) / len(dataset.data),
                    "execution_time_ms": result.execution_time_ms,
                    "threshold": result.threshold,
                    "mean_score": sum(s.value for s in result.scores)
                    / len(result.scores),
                    "max_score": max(s.value for s in result.scores),
                    "min_score": min(s.value for s in result.scores),
                }

        # Calculate summary statistics
        if comparison["individual_results"]:
            all_times = [
                r["execution_time_ms"]
                for r in comparison["individual_results"].values()
            ]
            all_rates = [
                r["anomaly_rate"] for r in comparison["individual_results"].values()
            ]

            comparison["summary"] = {
                "fastest_detector": min(
                    comparison["individual_results"].items(),
                    key=lambda x: x[1]["execution_time_ms"],
                )[0],
                "slowest_detector": max(
                    comparison["individual_results"].items(),
                    key=lambda x: x[1]["execution_time_ms"],
                )[0],
                "avg_execution_time": sum(all_times) / len(all_times),
                "avg_anomaly_rate": sum(all_rates) / len(all_rates),
                "execution_time_std": float(np.std(all_times)),
                "anomaly_rate_std": float(np.std(all_rates)),
            }

            # Create rankings
            comparison["rankings"] = {
                "by_speed": sorted(
                    comparison["individual_results"].items(),
                    key=lambda x: x[1]["execution_time_ms"],
                ),
                "by_anomaly_count": sorted(
                    comparison["individual_results"].items(),
                    key=lambda x: x[1]["n_anomalies"],
                    reverse=True,
                ),
            }

        return comparison

    async def _optimize_ensemble_weights(
        self, ensemble: EnsembleMetaAdapter, dataset: Dataset
    ) -> None:
        """Optimize ensemble weights based on individual detector performance."""
        # This is a simplified weight optimization
        # In practice, you might use cross-validation or other techniques

        base_detectors = ensemble.base_detectors
        if len(base_detectors) <= 1:
            return

        # Run individual detections
        individual_results = []
        for detector in base_detectors:
            try:
                result = await self._detect_with_detector(detector, dataset)
                individual_results.append(result)
            except Exception:
                # Skip problematic detectors
                continue

        if len(individual_results) < 2:
            return

        # Calculate weights based on score variance (lower variance = higher weight)
        import numpy as np

        weights = {}
        for detector, result in zip(base_detectors, individual_results, strict=False):
            scores = [s.value for s in result.scores]
            # Lower variance indicates more consistent scoring
            variance = np.var(scores)
            # Inverse variance weighting
            weight = 1.0 / (
                variance + 1e-8
            )  # Add small value to avoid division by zero
            weights[detector.name] = weight

        # Normalize weights
        total_weight = sum(weights.values())
        for name in weights:
            weights[name] = weights[name] / total_weight * len(weights)

        # Update ensemble weights
        for detector in base_detectors:
            if detector.name in weights:
                ensemble.update_detector_weight(detector.name, weights[detector.name])

    def _track_performance(self, metrics: dict[str, Any]) -> None:
        """Track performance metrics."""
        metrics["timestamp"] = time.time()
        self._performance_history.append(metrics)

        # Keep only recent history (last 1000 entries)
        if len(self._performance_history) > 1000:
            self._performance_history = self._performance_history[-1000:]


# Import numpy for calculations
import numpy as np
