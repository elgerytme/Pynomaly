"""Use case for explainability operations in anomaly detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pynomaly.domain.entities import Dataset, Detector
from pynomaly.domain.services.explainability_service import (
    CohortExplanation,
    ExplainabilityService,
    ExplanationMethod,
    GlobalExplanation,
    LocalExplanation,
)

logger = logging.getLogger(__name__)


@dataclass
class ExplainPredictionRequest:
    """Request for explaining a single prediction."""

    detector_id: str
    instance_data: dict[str, Any] | np.ndarray | pd.Series
    explanation_method: ExplanationMethod = ExplanationMethod.SHAP
    background_dataset_id: str | None = None
    instance_id: str | None = None
    include_counterfactuals: bool = False
    max_features: int = 10


@dataclass
class ExplainModelRequest:
    """Request for explaining global model behavior."""

    detector_id: str
    dataset_id: str
    explanation_method: ExplanationMethod = ExplanationMethod.SHAP
    sample_size: int = 1000
    include_interactions: bool = False
    feature_groups: dict[str, list[str]] | None = None


@dataclass
class ExplainCohortRequest:
    """Request for explaining a cohort of similar instances."""

    detector_id: str
    dataset_id: str
    instance_indices: list[int]
    explanation_method: ExplanationMethod = ExplanationMethod.SHAP
    cohort_name: str | None = None
    similarity_threshold: float = 0.8


@dataclass
class CompareExplanationsRequest:
    """Request for comparing explanations across different methods."""

    detector_id: str
    instance_data: dict[str, Any] | np.ndarray | pd.Series
    explanation_methods: list[ExplanationMethod]
    background_dataset_id: str | None = None
    consistency_analysis: bool = True


@dataclass
class ExplainabilityResponse:
    """Response for explainability operations."""

    success: bool
    explanation: None | (LocalExplanation | GlobalExplanation | CohortExplanation) = (
        None
    )
    explanations: dict[str, Any] | None = None  # For multiple explanations
    metadata: dict[str, Any] | None = None
    error_message: str | None = None
    execution_time_seconds: float = 0.0


class ExplainabilityUseCase:
    """Use case for managing explainability operations."""

    def __init__(
        self,
        explainability_service: ExplainabilityService,
        detector_repository,
        dataset_repository,
        enable_caching: bool = True,
        cache_ttl_hours: int = 24,
    ):
        """Initialize explainability use case.

        Args:
            explainability_service: Domain explainability service
            detector_repository: Repository for detector access
            dataset_repository: Repository for dataset access
            enable_caching: Whether to enable explanation caching
            cache_ttl_hours: Cache time-to-live in hours
        """
        self.explainability_service = explainability_service
        self.detector_repository = detector_repository
        self.dataset_repository = dataset_repository
        self.enable_caching = enable_caching
        self.cache_ttl_hours = cache_ttl_hours
        self._explanation_cache: dict[str, Any] = {}

    async def explain_prediction(
        self, request: ExplainPredictionRequest
    ) -> ExplainabilityResponse:
        """Explain a single prediction with enhanced features.

        Args:
            request: Explanation request with prediction details

        Returns:
            Response containing local explanation and metadata
        """
        try:
            start_time = logger.time() if hasattr(logger, "time") else 0
            logger.info(f"Explaining prediction for detector {request.detector_id}")

            # Validate detector
            detector = await self.detector_repository.get(request.detector_id)
            if not detector:
                return ExplainabilityResponse(
                    success=False,
                    error_message=f"Detector {request.detector_id} not found",
                )

            if not detector.is_fitted:
                return ExplainabilityResponse(
                    success=False,
                    error_message=f"Detector {request.detector_id} is not trained",
                )

            # Prepare instance data
            feature_names = await self._get_feature_names(
                detector, request.background_dataset_id
            )
            instance_array, feature_names = self._prepare_instance_data(
                request.instance_data, feature_names
            )

            # Get background data if specified
            background_data = None
            if request.background_dataset_id:
                background_data = await self._get_background_data(
                    request.background_dataset_id
                )

            # Check cache first
            cache_key = self._generate_cache_key(
                "prediction",
                request.detector_id,
                instance_array,
                request.explanation_method,
            )

            if self.enable_caching and cache_key in self._explanation_cache:
                cached_result = self._explanation_cache[cache_key]
                if self._is_cache_valid(cached_result["timestamp"]):
                    logger.info("Returning cached explanation")
                    return ExplainabilityResponse(
                        success=True,
                        explanation=cached_result["explanation"],
                        metadata=cached_result.get("metadata", {}),
                        execution_time_seconds=0.0,
                    )

            # Generate explanation
            if request.explanation_method == ExplanationMethod.SHAP:
                explanation = await self._explain_with_shap(
                    detector, instance_array, feature_names, background_data
                )
            elif request.explanation_method == ExplanationMethod.LIME:
                explanation = await self._explain_with_lime(
                    detector, instance_array, feature_names, background_data
                )
            else:
                explanation = self.explainability_service.explain_instance(
                    instance=instance_array,
                    model=detector,
                    feature_names=feature_names,
                    method=request.explanation_method,
                )

            # Enhance explanation with additional metadata
            metadata = await self._generate_explanation_metadata(
                detector, explanation, request
            )

            # Add counterfactuals if requested
            if request.include_counterfactuals:
                counterfactuals = await self._generate_counterfactuals(
                    detector, instance_array, feature_names, explanation
                )
                metadata["counterfactuals"] = counterfactuals

            # Cache result
            if self.enable_caching:
                self._cache_explanation(cache_key, explanation, metadata)

            end_time = logger.time() if hasattr(logger, "time") else 0
            execution_time = (end_time - start_time) if start_time > 0 else 0.0

            return ExplainabilityResponse(
                success=True,
                explanation=explanation,
                metadata=metadata,
                execution_time_seconds=execution_time,
            )

        except Exception as e:
            logger.error(f"Error explaining prediction: {str(e)}")
            return ExplainabilityResponse(
                success=False, error_message=f"Failed to explain prediction: {str(e)}"
            )

    async def explain_model(
        self, request: ExplainModelRequest
    ) -> ExplainabilityResponse:
        """Explain global model behavior with advanced analysis.

        Args:
            request: Model explanation request

        Returns:
            Response containing global explanation and analysis
        """
        try:
            start_time = logger.time() if hasattr(logger, "time") else 0
            logger.info(f"Explaining model for detector {request.detector_id}")

            # Validate detector and dataset
            detector = await self.detector_repository.get(request.detector_id)
            if not detector:
                return ExplainabilityResponse(
                    success=False,
                    error_message=f"Detector {request.detector_id} not found",
                )

            dataset = await self.dataset_repository.get(request.dataset_id)
            if not dataset:
                return ExplainabilityResponse(
                    success=False,
                    error_message=f"Dataset {request.dataset_id} not found",
                )

            # Prepare data
            feature_names = list(dataset.features.columns)
            sample_data = self._sample_dataset(dataset, request.sample_size)

            # Check cache
            cache_key = self._generate_cache_key(
                "model",
                request.detector_id,
                request.dataset_id,
                request.explanation_method,
            )

            if self.enable_caching and cache_key in self._explanation_cache:
                cached_result = self._explanation_cache[cache_key]
                if self._is_cache_valid(cached_result["timestamp"]):
                    return ExplainabilityResponse(
                        success=True,
                        explanation=cached_result["explanation"],
                        metadata=cached_result.get("metadata", {}),
                    )

            # Generate global explanation
            explanation = self.explainability_service.explain_model(
                data=sample_data.values,
                model=detector,
                feature_names=feature_names,
                method=request.explanation_method,
                max_samples=request.sample_size,
            )

            # Enhance with additional analysis
            metadata = await self._generate_model_metadata(
                detector, dataset, explanation, request
            )

            # Add feature interactions if requested
            if request.include_interactions:
                interactions = await self._calculate_feature_interactions(
                    detector, sample_data, feature_names
                )
                metadata["feature_interactions"] = interactions

            # Add feature group analysis if specified
            if request.feature_groups:
                group_analysis = await self._analyze_feature_groups(
                    explanation, request.feature_groups
                )
                metadata["group_analysis"] = group_analysis

            # Cache result
            if self.enable_caching:
                self._cache_explanation(cache_key, explanation, metadata)

            end_time = logger.time() if hasattr(logger, "time") else 0
            execution_time = (end_time - start_time) if start_time > 0 else 0.0

            return ExplainabilityResponse(
                success=True,
                explanation=explanation,
                metadata=metadata,
                execution_time_seconds=execution_time,
            )

        except Exception as e:
            logger.error(f"Error explaining model: {str(e)}")
            return ExplainabilityResponse(
                success=False, error_message=f"Failed to explain model: {str(e)}"
            )

    async def explain_cohort(
        self, request: ExplainCohortRequest
    ) -> ExplainabilityResponse:
        """Explain behavior for a cohort of similar instances.

        Args:
            request: Cohort explanation request

        Returns:
            Response containing cohort explanation and patterns
        """
        try:
            start_time = logger.time() if hasattr(logger, "time") else 0
            logger.info(f"Explaining cohort for detector {request.detector_id}")

            # Validate inputs
            detector = await self.detector_repository.get(request.detector_id)
            if not detector:
                return ExplainabilityResponse(
                    success=False,
                    error_message=f"Detector {request.detector_id} not found",
                )

            dataset = await self.dataset_repository.get(request.dataset_id)
            if not dataset:
                return ExplainabilityResponse(
                    success=False,
                    error_message=f"Dataset {request.dataset_id} not found",
                )

            # Validate indices
            max_index = len(dataset.data) - 1
            invalid_indices = [i for i in request.instance_indices if i > max_index]
            if invalid_indices:
                return ExplainabilityResponse(
                    success=False, error_message=f"Invalid indices: {invalid_indices}"
                )

            # Extract cohort data
            cohort_data = dataset.data.iloc[request.instance_indices]
            feature_names = list(dataset.features.columns)
            cohort_id = (
                request.cohort_name
                or f"cohort_{request.detector_id}_{len(request.instance_indices)}"
            )

            # Generate cohort explanation
            explanation = self.explainability_service.explain_cohort(
                instances=cohort_data.values,
                model=detector,
                feature_names=feature_names,
                cohort_id=cohort_id,
                method=request.explanation_method,
            )

            # Enhance with cohort analysis
            metadata = await self._generate_cohort_metadata(
                detector, cohort_data, explanation, request
            )

            # Add similarity analysis
            if len(request.instance_indices) > 1:
                similarity_matrix = await self._calculate_cohort_similarity(
                    cohort_data, request.similarity_threshold
                )
                metadata["similarity_analysis"] = similarity_matrix

            end_time = logger.time() if hasattr(logger, "time") else 0
            execution_time = (end_time - start_time) if start_time > 0 else 0.0

            return ExplainabilityResponse(
                success=True,
                explanation=explanation,
                metadata=metadata,
                execution_time_seconds=execution_time,
            )

        except Exception as e:
            logger.error(f"Error explaining cohort: {str(e)}")
            return ExplainabilityResponse(
                success=False, error_message=f"Failed to explain cohort: {str(e)}"
            )

    async def compare_explanations(
        self, request: CompareExplanationsRequest
    ) -> ExplainabilityResponse:
        """Compare explanations across different methods.

        Args:
            request: Comparison request with methods to compare

        Returns:
            Response containing comparison analysis
        """
        try:
            start_time = logger.time() if hasattr(logger, "time") else 0
            logger.info(f"Comparing explanations for detector {request.detector_id}")

            # Validate detector
            detector = await self.detector_repository.get(request.detector_id)
            if not detector:
                return ExplainabilityResponse(
                    success=False,
                    error_message=f"Detector {request.detector_id} not found",
                )

            # Prepare data
            feature_names = await self._get_feature_names(
                detector, request.background_dataset_id
            )
            instance_array, feature_names = self._prepare_instance_data(
                request.instance_data, feature_names
            )

            # Generate explanations for each method
            explanations = {}
            errors = {}

            for method in request.explanation_methods:
                try:
                    if method in self.explainability_service.get_available_methods():
                        explanation = self.explainability_service.explain_instance(
                            instance=instance_array,
                            model=detector,
                            feature_names=feature_names,
                            method=method,
                        )
                        explanations[method.value] = explanation
                    else:
                        errors[method.value] = f"Method {method.value} not available"
                except Exception as e:
                    errors[method.value] = str(e)

            # Perform consistency analysis if requested
            metadata = {
                "comparison_methods": [m.value for m in request.explanation_methods],
                "successful_methods": list(explanations.keys()),
                "failed_methods": errors,
            }

            if request.consistency_analysis and len(explanations) > 1:
                consistency_analysis = await self._analyze_explanation_consistency(
                    explanations
                )
                metadata["consistency_analysis"] = consistency_analysis

            end_time = logger.time() if hasattr(logger, "time") else 0
            execution_time = (end_time - start_time) if start_time > 0 else 0.0

            return ExplainabilityResponse(
                success=True,
                explanations=explanations,
                metadata=metadata,
                execution_time_seconds=execution_time,
            )

        except Exception as e:
            logger.error(f"Error comparing explanations: {str(e)}")
            return ExplainabilityResponse(
                success=False, error_message=f"Failed to compare explanations: {str(e)}"
            )

    # Helper methods

    async def _get_feature_names(
        self, detector: Detector, dataset_id: str | None = None
    ) -> list[str]:
        """Get feature names for the detector."""
        if dataset_id:
            dataset = await self.dataset_repository.get(dataset_id)
            if dataset:
                return list(dataset.features.columns)

        # Fallback to detector metadata or default names
        if hasattr(detector, "feature_names") and detector.feature_names:
            return detector.feature_names

        # Generate default feature names
        n_features = getattr(detector, "n_features_", 10)  # Default assumption
        return [f"feature_{i}" for i in range(n_features)]

    def _prepare_instance_data(
        self,
        instance_data: dict[str, Any] | np.ndarray | pd.Series,
        feature_names: list[str],
    ) -> tuple[np.ndarray, list[str]]:
        """Prepare instance data for explanation."""
        if isinstance(instance_data, dict):
            # Ensure feature order matches feature_names
            values = [instance_data.get(name, 0.0) for name in feature_names]
            return np.array(values).reshape(1, -1), feature_names
        elif isinstance(instance_data, pd.Series):
            return instance_data.values.reshape(1, -1), list(instance_data.index)
        elif isinstance(instance_data, np.ndarray):
            if instance_data.ndim == 1:
                instance_data = instance_data.reshape(1, -1)
            return instance_data, feature_names[: instance_data.shape[1]]
        else:
            raise ValueError(f"Unsupported instance data type: {type(instance_data)}")

    async def _get_background_data(self, dataset_id: str) -> np.ndarray | None:
        """Get background data for explanation context."""
        dataset = await self.dataset_repository.get(dataset_id)
        if dataset:
            return dataset.features.values
        return None

    def _sample_dataset(self, dataset: Dataset, sample_size: int) -> pd.DataFrame:
        """Sample dataset for analysis."""
        if len(dataset.data) <= sample_size:
            return dataset.features

        return dataset.features.sample(n=sample_size, random_state=42)

    def _generate_cache_key(self, operation: str, *args) -> str:
        """Generate cache key for explanations."""
        import hashlib

        key_string = f"{operation}_{hash(str(args))}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if cached explanation is still valid."""
        import time

        cache_ttl_seconds = self.cache_ttl_hours * 3600
        return (time.time() - timestamp) < cache_ttl_seconds

    def _cache_explanation(
        self, cache_key: str, explanation: Any, metadata: dict[str, Any]
    ) -> None:
        """Cache explanation result."""
        import time

        self._explanation_cache[cache_key] = {
            "explanation": explanation,
            "metadata": metadata,
            "timestamp": time.time(),
        }

    async def _explain_with_shap(
        self,
        detector: Detector,
        instance: np.ndarray,
        feature_names: list[str],
        background_data: np.ndarray | None,
    ) -> LocalExplanation:
        """Generate SHAP explanation with optimizations."""
        # This could be enhanced with specific SHAP optimizations
        return self.explainability_service.explain_instance(
            instance=instance,
            model=detector,
            feature_names=feature_names,
            method=ExplanationMethod.SHAP,
        )

    async def _explain_with_lime(
        self,
        detector: Detector,
        instance: np.ndarray,
        feature_names: list[str],
        background_data: np.ndarray | None,
    ) -> LocalExplanation:
        """Generate LIME explanation with optimizations."""
        return self.explainability_service.explain_instance(
            instance=instance,
            model=detector,
            feature_names=feature_names,
            method=ExplanationMethod.LIME,
        )

    async def _generate_explanation_metadata(
        self,
        detector: Detector,
        explanation: LocalExplanation,
        request: ExplainPredictionRequest,
    ) -> dict[str, Any]:
        """Generate metadata for explanation."""
        return {
            "detector_algorithm": (
                detector.algorithm if hasattr(detector, "algorithm") else "unknown"
            ),
            "explanation_method": explanation.explanation_method.value,
            "confidence_score": explanation.confidence,
            "feature_count": len(explanation.feature_contributions),
            "top_contributing_features": [
                contrib.feature_name
                for contrib in explanation.feature_contributions[:3]
            ],
            "anomaly_score": explanation.anomaly_score,
            "prediction": explanation.prediction,
        }

    async def _generate_model_metadata(
        self,
        detector: Detector,
        dataset: Dataset,
        explanation: GlobalExplanation,
        request: ExplainModelRequest,
    ) -> dict[str, Any]:
        """Generate metadata for model explanation."""
        return {
            "detector_algorithm": (
                detector.algorithm if hasattr(detector, "algorithm") else "unknown"
            ),
            "dataset_shape": dataset.data.shape,
            "sample_size_used": request.sample_size,
            "feature_count": len(explanation.feature_importances),
            "top_important_features": explanation.top_features[:5],
            "explanation_coverage": len(explanation.feature_importances)
            / len(dataset.features.columns),
        }

    async def _generate_cohort_metadata(
        self,
        detector: Detector,
        cohort_data: pd.DataFrame,
        explanation: CohortExplanation,
        request: ExplainCohortRequest,
    ) -> dict[str, Any]:
        """Generate metadata for cohort explanation."""
        return {
            "cohort_size": len(cohort_data),
            "cohort_id": explanation.cohort_id,
            "feature_statistics": {
                "mean": cohort_data.mean().to_dict(),
                "std": cohort_data.std().to_dict(),
                "min": cohort_data.min().to_dict(),
                "max": cohort_data.max().to_dict(),
            },
            "common_feature_count": len(explanation.common_features),
        }

    async def _generate_counterfactuals(
        self,
        detector: Detector,
        instance: np.ndarray,
        feature_names: list[str],
        explanation: LocalExplanation,
    ) -> list[dict[str, Any]]:
        """Generate counterfactual explanations."""
        # Simplified counterfactual generation
        # In production, this would use specialized libraries like DiCE or Alibi
        counterfactuals = []

        # Find top contributing features
        top_features = sorted(
            explanation.feature_contributions,
            key=lambda x: abs(x.contribution),
            reverse=True,
        )[:3]

        for feature in top_features:
            # Simple counterfactual: flip the sign of contribution
            feature_idx = feature_names.index(feature.feature_name)
            counterfactual_instance = instance.copy()

            # Adjust feature value (simplified approach)
            if feature.contribution > 0:
                counterfactual_instance[
                    0, feature_idx
                ] *= 0.5  # Reduce positive contribution
            else:
                counterfactual_instance[
                    0, feature_idx
                ] *= -0.5  # Reduce negative contribution

            counterfactuals.append(
                {
                    "modified_feature": feature.feature_name,
                    "original_value": feature.value,
                    "counterfactual_value": float(
                        counterfactual_instance[0, feature_idx]
                    ),
                    "expected_impact": (
                        "Reduced anomaly score"
                        if feature.contribution > 0
                        else "Increased anomaly score"
                    ),
                }
            )

        return counterfactuals

    async def _calculate_feature_interactions(
        self, detector: Detector, data: pd.DataFrame, feature_names: list[str]
    ) -> dict[str, float]:
        """Calculate feature interaction effects."""
        # Simplified interaction calculation
        # In production, this would use more sophisticated methods
        interactions = {}

        for i, feat1 in enumerate(feature_names):
            for j, feat2 in enumerate(feature_names[i + 1 :], i + 1):
                # Simple correlation-based interaction measure
                if feat1 in data.columns and feat2 in data.columns:
                    correlation = data[feat1].corr(data[feat2])
                    interactions[f"{feat1}_x_{feat2}"] = abs(correlation)

        return interactions

    async def _analyze_feature_groups(
        self, explanation: GlobalExplanation, feature_groups: dict[str, list[str]]
    ) -> dict[str, dict[str, float]]:
        """Analyze feature importance by groups."""
        group_analysis = {}

        for group_name, features in feature_groups.items():
            group_importance = 0.0
            group_features_found = []

            for feature in features:
                if feature in explanation.feature_importances:
                    group_importance += explanation.feature_importances[feature]
                    group_features_found.append(feature)

            group_analysis[group_name] = {
                "total_importance": group_importance,
                "average_importance": (
                    group_importance / len(group_features_found)
                    if group_features_found
                    else 0.0
                ),
                "feature_count": len(group_features_found),
                "features": group_features_found,
            }

        return group_analysis

    async def _calculate_cohort_similarity(
        self, cohort_data: pd.DataFrame, threshold: float
    ) -> dict[str, Any]:
        """Calculate similarity matrix for cohort instances."""
        from scipy.spatial.distance import pdist, squareform

        # Calculate pairwise distances
        distances = pdist(cohort_data.values, metric="euclidean")
        distance_matrix = squareform(distances)

        # Convert to similarity (inverse of distance)
        max_distance = np.max(distance_matrix)
        similarity_matrix = (
            1 - (distance_matrix / max_distance)
            if max_distance > 0
            else np.ones_like(distance_matrix)
        )

        # Find highly similar pairs
        similar_pairs = []
        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix)):
                if similarity_matrix[i, j] >= threshold:
                    similar_pairs.append(
                        {
                            "instance_1": i,
                            "instance_2": j,
                            "similarity": float(similarity_matrix[i, j]),
                        }
                    )

        return {
            "similarity_matrix": similarity_matrix.tolist(),
            "average_similarity": float(
                np.mean(similarity_matrix[np.triu_indices(len(similarity_matrix), k=1)])
            ),
            "high_similarity_pairs": similar_pairs,
            "cohort_compactness": float(np.mean(similarity_matrix)),
        }

    async def _analyze_explanation_consistency(
        self, explanations: dict[str, LocalExplanation]
    ) -> dict[str, Any]:
        """Analyze consistency across different explanation methods."""
        if len(explanations) < 2:
            return {"error": "Need at least 2 explanations for consistency analysis"}

        # Extract feature rankings from each explanation
        feature_rankings = {}
        for method, explanation in explanations.items():
            rankings = {
                contrib.feature_name: contrib.rank
                for contrib in explanation.feature_contributions
            }
            feature_rankings[method] = rankings

        # Calculate rank correlation between methods
        methods = list(explanations.keys())
        correlations = {}

        for i, method1 in enumerate(methods):
            for method2 in methods[i + 1 :]:
                # Get common features
                common_features = set(feature_rankings[method1].keys()) & set(
                    feature_rankings[method2].keys()
                )

                if len(common_features) > 1:
                    ranks1 = [feature_rankings[method1][f] for f in common_features]
                    ranks2 = [feature_rankings[method2][f] for f in common_features]

                    # Calculate Spearman correlation
                    from scipy.stats import spearmanr

                    correlation, p_value = spearmanr(ranks1, ranks2)

                    correlations[f"{method1}_vs_{method2}"] = {
                        "correlation": float(correlation),
                        "p_value": float(p_value),
                        "common_features": len(common_features),
                    }

        # Calculate overall consistency score
        correlation_values = [
            corr["correlation"]
            for corr in correlations.values()
            if not np.isnan(corr["correlation"])
        ]
        overall_consistency = np.mean(correlation_values) if correlation_values else 0.0

        return {
            "pairwise_correlations": correlations,
            "overall_consistency_score": float(overall_consistency),
            "consistency_interpretation": self._interpret_consistency(
                overall_consistency
            ),
        }

    def _interpret_consistency(self, consistency_score: float) -> str:
        """Interpret consistency score."""
        if consistency_score >= 0.8:
            return "High consistency - explanations are very similar"
        elif consistency_score >= 0.6:
            return "Moderate consistency - explanations have some agreement"
        elif consistency_score >= 0.4:
            return "Low consistency - explanations differ significantly"
        else:
            return "Very low consistency - explanations are inconsistent"
