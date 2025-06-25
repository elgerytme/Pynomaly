"""Application service for explainability operations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from pynomaly.domain.services.explainability_service import (
    CohortExplanation,
)
from pynomaly.domain.services.explainability_service import (
    ExplainabilityService as DomainExplainabilityService,
)
from pynomaly.domain.services.explainability_service import (
    ExplanationMethod,
    GlobalExplanation,
    LocalExplanation,
)
from pynomaly.infrastructure.repositories.dataset_repository import DatasetRepository
from pynomaly.infrastructure.repositories.detector_repository import DetectorRepository

logger = logging.getLogger(__name__)


@dataclass
class ExplanationRequest:
    """Request for explanation generation."""

    detector_id: str
    dataset_id: str | None = None
    instance_data: dict[str, Any] | None = None
    instance_indices: list[int] | None = None
    explanation_method: ExplanationMethod = ExplanationMethod.SHAP
    max_features: int = 10
    background_samples: int = 100


@dataclass
class ExplanationResponse:
    """Response from explanation generation."""

    success: bool
    explanation: LocalExplanation | GlobalExplanation | CohortExplanation | None = None
    message: str = ""
    error: str | None = None
    execution_time: float = 0.0


class ApplicationExplainabilityService:
    """Application service for managing explainability operations."""

    def __init__(
        self,
        domain_explainability_service: DomainExplainabilityService,
        detector_repository: DetectorRepository,
        dataset_repository: DatasetRepository,
    ):
        """Initialize application explainability service.

        Args:
            domain_explainability_service: Domain explainability service
            detector_repository: Repository for detector management
            dataset_repository: Repository for dataset management
        """
        self.domain_service = domain_explainability_service
        self.detector_repository = detector_repository
        self.dataset_repository = dataset_repository

    async def explain_instance(
        self, request: ExplanationRequest
    ) -> ExplanationResponse:
        """Generate explanation for a single instance."""
        try:
            start_time = logger.time() if hasattr(logger, "time") else 0

            # Get detector
            detector = await self.detector_repository.get_by_id(request.detector_id)
            if not detector:
                return ExplanationResponse(
                    success=False,
                    error=f"Detector not found: {request.detector_id}",
                    message="Failed to find detector",
                )

            # Get trained model
            if not detector.is_trained:
                return ExplanationResponse(
                    success=False,
                    error="Detector is not trained",
                    message="Detector must be trained before generating explanations",
                )

            # Prepare instance data
            if request.instance_data:
                # Convert dict to array
                feature_names = list(request.instance_data.keys())
                instance = np.array(
                    [request.instance_data[name] for name in feature_names]
                )
            elif request.dataset_id and request.instance_indices:
                # Get from dataset
                dataset = await self.dataset_repository.get_by_id(request.dataset_id)
                if not dataset:
                    return ExplanationResponse(
                        success=False,
                        error=f"Dataset not found: {request.dataset_id}",
                        message="Failed to find dataset",
                    )

                # Get instance from dataset
                if not request.instance_indices or request.instance_indices[0] >= len(
                    dataset.data
                ):
                    return ExplanationResponse(
                        success=False,
                        error="Invalid instance index",
                        message="Instance index out of range",
                    )

                instance = dataset.data.iloc[request.instance_indices[0]].values
                feature_names = dataset.data.columns.tolist()
            else:
                return ExplanationResponse(
                    success=False,
                    error="Either instance_data or dataset_id with instance_indices required",
                    message="Insufficient data for explanation",
                )

            # Generate explanation
            explanation = self.domain_service.explain_instance(
                instance=instance,
                model=detector.model,
                feature_names=feature_names,
                method=request.explanation_method,
                max_features=request.max_features,
            )

            end_time = logger.time() if hasattr(logger, "time") else 0
            execution_time = end_time - start_time if start_time > 0 else 0.0

            return ExplanationResponse(
                success=True,
                explanation=explanation,
                message=f"Generated {request.explanation_method.value} explanation successfully",
                execution_time=execution_time,
            )

        except Exception as e:
            logger.error(f"Failed to generate instance explanation: {e}")
            return ExplanationResponse(
                success=False, error=str(e), message="Failed to generate explanation"
            )

    async def explain_model(self, request: ExplanationRequest) -> ExplanationResponse:
        """Generate global explanation for the model."""
        try:
            start_time = logger.time() if hasattr(logger, "time") else 0

            # Get detector
            detector = await self.detector_repository.get_by_id(request.detector_id)
            if not detector:
                return ExplanationResponse(
                    success=False,
                    error=f"Detector not found: {request.detector_id}",
                    message="Failed to find detector",
                )

            # Get trained model
            if not detector.is_trained:
                return ExplanationResponse(
                    success=False,
                    error="Detector is not trained",
                    message="Detector must be trained before generating explanations",
                )

            # Get dataset for background data
            if not request.dataset_id:
                return ExplanationResponse(
                    success=False,
                    error="Dataset ID required for global explanation",
                    message="Global explanations require background data",
                )

            dataset = await self.dataset_repository.get_by_id(request.dataset_id)
            if not dataset:
                return ExplanationResponse(
                    success=False,
                    error=f"Dataset not found: {request.dataset_id}",
                    message="Failed to find dataset",
                )

            # Prepare data
            data = dataset.data.values
            feature_names = dataset.data.columns.tolist()

            # Sample background data if needed
            if len(data) > request.background_samples:
                indices = np.random.choice(
                    len(data), request.background_samples, replace=False
                )
                background_data = data[indices]
            else:
                background_data = data

            # Generate explanation
            explanation = self.domain_service.explain_model(
                data=background_data,
                model=detector.model,
                feature_names=feature_names,
                method=request.explanation_method,
                max_samples=request.background_samples,
            )

            end_time = logger.time() if hasattr(logger, "time") else 0
            execution_time = end_time - start_time if start_time > 0 else 0.0

            return ExplanationResponse(
                success=True,
                explanation=explanation,
                message=f"Generated global {request.explanation_method.value} explanation successfully",
                execution_time=execution_time,
            )

        except Exception as e:
            logger.error(f"Failed to generate global explanation: {e}")
            return ExplanationResponse(
                success=False,
                error=str(e),
                message="Failed to generate global explanation",
            )

    async def explain_cohort(self, request: ExplanationRequest) -> ExplanationResponse:
        """Generate explanation for a cohort of instances."""
        try:
            start_time = logger.time() if hasattr(logger, "time") else 0

            # Get detector
            detector = await self.detector_repository.get_by_id(request.detector_id)
            if not detector:
                return ExplanationResponse(
                    success=False,
                    error=f"Detector not found: {request.detector_id}",
                    message="Failed to find detector",
                )

            # Get trained model
            if not detector.is_trained:
                return ExplanationResponse(
                    success=False,
                    error="Detector is not trained",
                    message="Detector must be trained before generating explanations",
                )

            # Get dataset and instances
            if not request.dataset_id or not request.instance_indices:
                return ExplanationResponse(
                    success=False,
                    error="Dataset ID and instance indices required for cohort explanation",
                    message="Cohort explanations require specific instances",
                )

            dataset = await self.dataset_repository.get_by_id(request.dataset_id)
            if not dataset:
                return ExplanationResponse(
                    success=False,
                    error=f"Dataset not found: {request.dataset_id}",
                    message="Failed to find dataset",
                )

            # Validate indices
            invalid_indices = [
                i for i in request.instance_indices if i >= len(dataset.data)
            ]
            if invalid_indices:
                return ExplanationResponse(
                    success=False,
                    error=f"Invalid instance indices: {invalid_indices}",
                    message="Some instance indices are out of range",
                )

            # Get cohort instances
            instances = dataset.data.iloc[request.instance_indices].values
            feature_names = dataset.data.columns.tolist()
            cohort_id = f"cohort_{request.dataset_id}_{len(request.instance_indices)}"

            # Generate explanation
            explanation = self.domain_service.explain_cohort(
                instances=instances,
                model=detector.model,
                feature_names=feature_names,
                cohort_id=cohort_id,
                method=request.explanation_method,
            )

            end_time = logger.time() if hasattr(logger, "time") else 0
            execution_time = end_time - start_time if start_time > 0 else 0.0

            return ExplanationResponse(
                success=True,
                explanation=explanation,
                message=f"Generated cohort {request.explanation_method.value} explanation successfully",
                execution_time=execution_time,
            )

        except Exception as e:
            logger.error(f"Failed to generate cohort explanation: {e}")
            return ExplanationResponse(
                success=False,
                error=str(e),
                message="Failed to generate cohort explanation",
            )

    async def compare_explanations(
        self, request: ExplanationRequest, methods: list[ExplanationMethod]
    ) -> dict[str, ExplanationResponse]:
        """Compare explanations from multiple methods."""
        try:
            results = {}

            for method in methods:
                method_request = ExplanationRequest(
                    detector_id=request.detector_id,
                    dataset_id=request.dataset_id,
                    instance_data=request.instance_data,
                    instance_indices=request.instance_indices,
                    explanation_method=method,
                    max_features=request.max_features,
                    background_samples=request.background_samples,
                )

                response = await self.explain_instance(method_request)
                results[method.value] = response

            return results

        except Exception as e:
            logger.error(f"Failed to compare explanations: {e}")
            error_response = ExplanationResponse(
                success=False, error=str(e), message="Failed to compare explanations"
            )
            return {method.value: error_response for method in methods}

    def get_available_methods(self) -> list[ExplanationMethod]:
        """Get list of available explanation methods."""
        return self.domain_service.get_available_methods()

    async def get_feature_statistics(
        self,
        detector_id: str,
        dataset_id: str,
        method: ExplanationMethod = ExplanationMethod.SHAP,
        sample_size: int = 100,
    ) -> dict[str, Any]:
        """Get feature statistics across multiple explanations."""
        try:
            # Get detector and dataset
            detector = await self.detector_repository.get_by_id(detector_id)
            dataset = await self.dataset_repository.get_by_id(dataset_id)

            if not detector or not dataset:
                return {"error": "Detector or dataset not found"}

            if not detector.is_trained:
                return {"error": "Detector is not trained"}

            # Sample instances
            n_samples = min(sample_size, len(dataset.data))
            indices = np.random.choice(len(dataset.data), n_samples, replace=False)

            # Generate explanations for sampled instances
            explanations = []
            for idx in indices:
                try:
                    request = ExplanationRequest(
                        detector_id=detector_id,
                        dataset_id=dataset_id,
                        instance_indices=[int(idx)],
                        explanation_method=method,
                    )
                    response = await self.explain_instance(request)
                    if response.success and response.explanation:
                        explanations.append(response.explanation)
                except Exception as e:
                    logger.warning(
                        f"Failed to generate explanation for instance {idx}: {e}"
                    )
                    continue

            if not explanations:
                return {"error": "Failed to generate any explanations"}

            # Calculate statistics
            stats = self.domain_service.get_feature_statistics(explanations)

            # Add top features ranking
            top_features = self.domain_service.rank_features_by_importance(explanations)

            return {
                "feature_statistics": stats,
                "top_features": top_features,
                "total_explanations": len(explanations),
                "method": method.value,
            }

        except Exception as e:
            logger.error(f"Failed to get feature statistics: {e}")
            return {"error": str(e)}
