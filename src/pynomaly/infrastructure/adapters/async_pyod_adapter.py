"""Async-enabled PyOD adapter with performance optimizations."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
from memory_profiler import profile

from pynomaly.domain.entities import Anomaly, Dataset, DetectionResult
from pynomaly.domain.exceptions import (
    DetectorNotFittedError,
    FittingError,
    InvalidAlgorithmError,
)
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate
from pynomaly.infrastructure.adapters.enhanced_pyod_adapter import (
    EnhancedPyODAdapter,
    AlgorithmMetadata,
)
from pynomaly.infrastructure.performance.profiling_service import get_profiling_service, ProfilerType
from pynomaly.infrastructure.cache.redis_cache import cached


class AsyncPyODAdapter(EnhancedPyODAdapter):
    """Async-enabled PyOD adapter with performance optimizations."""

    def __init__(
        self,
        algorithm_name: str,
        name: str | None = None,
        contamination_rate: ContaminationRate | None = None,
        enable_profiling: bool = False,
        **kwargs: Any,
    ):
        """Initialize async PyOD adapter.

        Args:
            algorithm_name: Name of the PyOD algorithm
            name: Optional custom name
            contamination_rate: Expected contamination rate
            enable_profiling: Whether to enable performance profiling
            **kwargs: Algorithm-specific parameters
        """
        super().__init__(algorithm_name, name, contamination_rate, **kwargs)
        self._profiler = get_profiling_service() if enable_profiling else None
        self._enable_profiling = enable_profiling

    async def fit_async(self, dataset: Dataset) -> None:
        """Fit the PyOD detector asynchronously.

        Args:
            dataset: Dataset to fit on

        Raises:
            FittingError: If fitting fails
        """
        if self._enable_profiling and self._profiler:
            with self._profiler.profile_context(f"fit_async_{self._algorithm_name}"):
                await asyncio.to_thread(self._fit_optimized, dataset)
        else:
            await asyncio.to_thread(self._fit_optimized, dataset)

    async def detect_async(self, dataset: Dataset) -> DetectionResult:
        """Detect anomalies asynchronously.

        Args:
            dataset: Dataset to analyze

        Returns:
            Detection result with anomalies, scores, and metadata

        Raises:
            DetectorNotFittedError: If detector is not fitted
        """
        if not self._is_fitted:
            raise DetectorNotFittedError(detector_name=self._name, operation="detect")

        if self._enable_profiling and self._profiler:
            with self._profiler.profile_context(f"detect_async_{self._algorithm_name}"):
                result = await asyncio.to_thread(self._detect_optimized, dataset)
                return result
        else:
            result = await asyncio.to_thread(self._detect_optimized, dataset)
            return result

    async def score_async(self, dataset: Dataset) -> list[AnomalyScore]:
        """Calculate anomaly scores asynchronously.

        Args:
            dataset: Dataset to score

        Returns:
            List of anomaly scores
        """
        if not self._is_fitted:
            raise DetectorNotFittedError(detector_name=self._name, operation="score")

        if self._enable_profiling and self._profiler:
            with self._profiler.profile_context(f"score_async_{self._algorithm_name}"):
                scores = await asyncio.to_thread(self._score_optimized, dataset)
                return scores
        else:
            scores = await asyncio.to_thread(self._score_optimized, dataset)
            return scores

    async def fit_detect_async(self, dataset: Dataset) -> DetectionResult:
        """Fit detector and detect anomalies asynchronously.

        Args:
            dataset: Dataset to fit and analyze

        Returns:
            Detection result
        """
        await self.fit_async(dataset)
        return await self.detect_async(dataset)

    @profile
    def _fit_optimized(self, dataset: Dataset) -> None:
        """Optimized fit method with profiling."""
        start_time = time.perf_counter()

        try:
            # Prepare data with vectorized operations
            X, feature_names = self._prepare_features_optimized(dataset)
            self._feature_names = feature_names

            # Initialize model with parameters
            model_params = self._prepare_model_parameters()
            self._model = self._model_class(**model_params)

            # Fit model
            self._model.fit(X)

            # Update state
            self._is_fitted = True
            training_time = time.perf_counter() - start_time

            # Store training metadata
            self._training_metadata = {
                "training_time_seconds": training_time,
                "n_samples": len(X),
                "n_features": X.shape[1],
                "feature_names": feature_names,
                "algorithm": self._algorithm_name,
                "parameters": model_params,
                "pyod_version": self._get_pyod_version(),
            }

        except Exception as e:
            raise FittingError(
                detector_name=self._name, reason=str(e), dataset_name=dataset.name
            ) from e

    @profile
    def _detect_optimized(self, dataset: Dataset) -> DetectionResult:
        """Optimized detect method with profiling."""
        start_time = time.perf_counter()

        try:
            # Prepare data with vectorized operations
            X, _ = self._prepare_features_optimized(dataset)

            # Get predictions and scores in batch
            labels = self._model.predict(X)  # 0=normal, 1=anomaly
            raw_scores = self._model.decision_function(X)

            # Vectorized score normalization
            normalized_scores = self._normalize_scores_vectorized(raw_scores)

            # Create anomaly score objects efficiently
            anomaly_scores = [
                AnomalyScore(value=float(score), method=f"pyod_{self._algorithm_name}")
                for score in normalized_scores
            ]

            # Create anomaly entities efficiently
            anomalies = self._create_anomaly_entities_optimized(
                dataset, labels, anomaly_scores, raw_scores
            )

            # Calculate threshold
            threshold = self._calculate_threshold_optimized(normalized_scores)

            execution_time = (time.perf_counter() - start_time) * 1000

            # Create result
            return DetectionResult(
                detector_id=hash(self._name),
                dataset_id=dataset.id,
                anomalies=anomalies,
                scores=anomaly_scores,
                labels=labels.tolist(),
                threshold=threshold,
                execution_time_ms=execution_time,
                metadata={
                    "algorithm": self._algorithm_name,
                    "category": self._metadata.category,
                    "detection_time_seconds": execution_time / 1000,
                    "n_anomalies": len(anomalies),
                    "contamination_rate": self._contamination_rate.value,
                    **self._training_metadata,
                },
            )

        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            return DetectionResult(
                detector_id=hash(self._name),
                dataset_id=dataset.id,
                anomalies=[],
                scores=[],
                labels=[],
                threshold=0.5,
                execution_time_ms=execution_time,
                metadata={
                    "algorithm": self._algorithm_name,
                    "error": str(e),
                    "status": "failed",
                },
            )

    @profile
    def _score_optimized(self, dataset: Dataset) -> list[AnomalyScore]:
        """Optimized score method with profiling."""
        X, _ = self._prepare_features_optimized(dataset)
        raw_scores = self._model.decision_function(X)
        normalized_scores = self._normalize_scores_vectorized(raw_scores)

        return [
            AnomalyScore(value=float(score), method=f"pyod_{self._algorithm_name}")
            for score in normalized_scores
        ]

    def _prepare_features_optimized(self, dataset: Dataset) -> tuple[np.ndarray, list[str]]:
        """Optimized feature preparation using vectorized operations."""
        # Use vectorized operations for better performance
        numeric_data = dataset.data.select_dtypes(include=[np.number])

        if numeric_data.empty:
            raise ValueError("No numeric features found in dataset")

        # Vectorized handling of missing values - avoid pandas copies
        numeric_data = numeric_data.fillna(numeric_data.mean())

        # Get feature names
        feature_names = numeric_data.columns.tolist()

        # Convert to numpy array with optimized memory layout
        return np.ascontiguousarray(numeric_data.values), feature_names

    def _normalize_scores_vectorized(self, scores: np.ndarray) -> np.ndarray:
        """Vectorized score normalization for better performance."""
        if len(scores) == 0:
            return scores

        # Use vectorized operations
        min_score = np.min(scores)
        max_score = np.max(scores)

        # Avoid division by zero
        if max_score == min_score:
            return np.full_like(scores, 0.5)

        # Vectorized normalization
        return (scores - min_score) / (max_score - min_score)

    def _create_anomaly_entities_optimized(
        self,
        dataset: Dataset,
        labels: np.ndarray,
        anomaly_scores: list[AnomalyScore],
        raw_scores: np.ndarray,
    ) -> list[Anomaly]:
        """Optimized anomaly entity creation."""
        anomalies = []
        
        # Find anomaly indices using vectorized operations
        anomaly_indices = np.where(labels == 1)[0]
        
        # Batch create anomalies
        for idx in anomaly_indices:
            anomaly = Anomaly(
                score=anomaly_scores[idx],
                data_point=dataset.data.iloc[idx].to_dict(),
                detector_name=self._name,
            )
            # Add algorithm-specific metadata
            anomaly.add_metadata("raw_score", float(raw_scores[idx]))
            anomaly.add_metadata("algorithm", self._algorithm_name)
            anomalies.append(anomaly)

        return anomalies

    def _calculate_threshold_optimized(self, scores: np.ndarray) -> float:
        """Optimized threshold calculation."""
        if len(scores) == 0:
            return 0.5
        
        # Use vectorized percentile calculation
        contamination_rate = self._contamination_rate.value
        percentile = (1 - contamination_rate) * 100
        return float(np.percentile(scores, percentile))

    def _get_pyod_version(self) -> str:
        """Get PyOD version."""
        try:
            import pyod
            return pyod.__version__
        except ImportError:
            return "unknown"

    @cached(ttl=3600)  # Cache for 1 hour
    def get_algorithm_metadata(self) -> dict[str, Any]:
        """Get algorithm metadata with caching."""
        return {
            "algorithm": self._algorithm_name,
            "category": self._metadata.category,
            "complexity_time": self._metadata.complexity_time,
            "complexity_space": self._metadata.complexity_space,
            "supports_streaming": self._metadata.supports_streaming,
            "supports_multivariate": self._metadata.supports_multivariate,
            "requires_gpu": self._metadata.requires_gpu,
            "description": self._metadata.description,
            "paper_reference": self._metadata.paper_reference,
            "typical_use_cases": self._metadata.typical_use_cases,
        }

    @cached(ttl=1800)  # Cache for 30 minutes
    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics with caching."""
        if self._profiler:
            return self._profiler.get_performance_summary()
        return {}

    def enable_profiling(self, enable: bool = True) -> None:
        """Enable or disable profiling."""
        self._enable_profiling = enable
        if enable:
            self._profiler = get_profiling_service()
        else:
            self._profiler = None
