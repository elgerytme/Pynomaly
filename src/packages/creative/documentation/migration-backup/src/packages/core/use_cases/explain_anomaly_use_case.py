"""Use case for explaining anomalies."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from pynomaly.application.services.explainability_service import (
    ApplicationExplainabilityService,
    ExplanationRequest,
)
from pynomaly.domain.services.explainability_service import ExplanationMethod

logger = logging.getLogger(__name__)


@dataclass
class ExplainAnomalyRequest:
    """Request for explaining anomalies."""

    detector_id: str
    dataset_id: str | None = None
    instance_data: dict[str, Any] | None = None
    anomaly_indices: list[int] | None = None
    explanation_method: str = "shap"
    max_features: int = 10
    include_cohort_analysis: bool = False
    compare_methods: bool = False


@dataclass
class ExplainAnomalyResponse:
    """Response from anomaly explanation."""

    success: bool
    explanations: dict[str, Any] = None
    feature_rankings: list[tuple[str, float]] | None = None
    cohort_analysis: dict[str, Any] | None = None
    method_comparison: dict[str, Any] | None = None
    message: str = ""
    error: str | None = None
    execution_time: float = 0.0


class ExplainAnomalyUseCase:
    """Use case for explaining anomaly detection results."""

    def __init__(self, explainability_service: ApplicationExplainabilityService):
        """Initialize use case.

        Args:
            explainability_service: Application explainability service
        """
        self.explainability_service = explainability_service

    async def execute(self, request: ExplainAnomalyRequest) -> ExplainAnomalyResponse:
        """Execute anomaly explanation.

        Args:
            request: Explanation request

        Returns:
            Explanation response with results
        """
        try:
            logger.info(
                f"Starting anomaly explanation for detector {request.detector_id}"
            )

            # Map method string to enum
            method_map = {
                "shap": ExplanationMethod.SHAP,
                "lime": ExplanationMethod.LIME,
                "feature_importance": ExplanationMethod.FEATURE_IMPORTANCE,
                "permutation_importance": ExplanationMethod.PERMUTATION_IMPORTANCE,
                "partial_dependence": ExplanationMethod.PARTIAL_DEPENDENCE,
            }

            method = method_map.get(
                request.explanation_method.lower(), ExplanationMethod.SHAP
            )
            explanations = {}

            # Single instance explanation
            if request.instance_data:
                exp_request = ExplanationRequest(
                    detector_id=request.detector_id,
                    instance_data=request.instance_data,
                    explanation_method=method,
                    max_features=request.max_features,
                )

                response = await self.explainability_service.explain_instance(
                    exp_request
                )
                if response.success:
                    explanations["instance"] = {
                        "explanation": response.explanation,
                        "execution_time": response.execution_time,
                    }
                else:
                    return ExplainAnomalyResponse(
                        success=False,
                        error=response.error,
                        message="Failed to explain instance",
                    )

            # Multiple anomaly instances
            if request.anomaly_indices and request.dataset_id:
                instance_explanations = []

                for idx in request.anomaly_indices:
                    exp_request = ExplanationRequest(
                        detector_id=request.detector_id,
                        dataset_id=request.dataset_id,
                        instance_indices=[idx],
                        explanation_method=method,
                        max_features=request.max_features,
                    )

                    response = await self.explainability_service.explain_instance(
                        exp_request
                    )
                    if response.success:
                        instance_explanations.append(
                            {
                                "index": idx,
                                "explanation": response.explanation,
                                "execution_time": response.execution_time,
                            }
                        )
                    else:
                        logger.warning(
                            f"Failed to explain instance {idx}: {response.error}"
                        )

                if instance_explanations:
                    explanations["anomalies"] = instance_explanations

            # Global model explanation
            if request.dataset_id:
                exp_request = ExplanationRequest(
                    detector_id=request.detector_id,
                    dataset_id=request.dataset_id,
                    explanation_method=method,
                    max_features=request.max_features,
                )

                response = await self.explainability_service.explain_model(exp_request)
                if response.success:
                    explanations["global"] = {
                        "explanation": response.explanation,
                        "execution_time": response.execution_time,
                    }

            # Cohort analysis
            cohort_analysis = None
            if (
                request.include_cohort_analysis
                and request.anomaly_indices
                and request.dataset_id
            ):
                exp_request = ExplanationRequest(
                    detector_id=request.detector_id,
                    dataset_id=request.dataset_id,
                    instance_indices=request.anomaly_indices,
                    explanation_method=method,
                    max_features=request.max_features,
                )

                response = await self.explainability_service.explain_cohort(exp_request)
                if response.success:
                    cohort_analysis = {
                        "explanation": response.explanation,
                        "execution_time": response.execution_time,
                    }

            # Method comparison
            method_comparison = None
            if request.compare_methods:
                available_methods = self.explainability_service.get_available_methods()
                comparison_methods = [m for m in available_methods if m != method][
                    :2
                ]  # Limit to 2 additional methods

                if comparison_methods and (
                    request.instance_data or request.anomaly_indices
                ):
                    exp_request = ExplanationRequest(
                        detector_id=request.detector_id,
                        dataset_id=request.dataset_id,
                        instance_data=request.instance_data,
                        instance_indices=(
                            request.anomaly_indices[:1]
                            if request.anomaly_indices
                            else None
                        ),
                        max_features=request.max_features,
                    )

                    comparison_results = (
                        await self.explainability_service.compare_explanations(
                            exp_request, [method] + comparison_methods
                        )
                    )
                    method_comparison = {
                        method_name: {
                            "success": result.success,
                            "explanation": (
                                result.explanation if result.success else None
                            ),
                            "error": result.error,
                            "execution_time": result.execution_time,
                        }
                        for method_name, result in comparison_results.items()
                    }

            # Get feature rankings if we have explanations
            feature_rankings = None
            if explanations:
                try:
                    stats = await self.explainability_service.get_feature_statistics(
                        detector_id=request.detector_id,
                        dataset_id=request.dataset_id or "",
                        method=method,
                        sample_size=50,
                    )
                    if "top_features" in stats:
                        feature_rankings = stats["top_features"]
                except Exception as e:
                    logger.warning(f"Failed to get feature rankings: {e}")

            if not explanations:
                return ExplainAnomalyResponse(
                    success=False,
                    error="No explanations could be generated",
                    message="Failed to generate any explanations",
                )

            return ExplainAnomalyResponse(
                success=True,
                explanations=explanations,
                feature_rankings=feature_rankings,
                cohort_analysis=cohort_analysis,
                method_comparison=method_comparison,
                message=f"Generated explanations using {request.explanation_method} method",
            )

        except Exception as e:
            logger.error(f"Failed to explain anomalies: {e}")
            return ExplainAnomalyResponse(
                success=False, error=str(e), message="Anomaly explanation failed"
            )
