"""Use case for explaining anomalies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

import numpy as np
import pandas as pd

from pynomaly.domain.entities import Dataset
from pynomaly.shared.protocols import (
    DetectorRepositoryProtocol,
    ExplainableDetectorProtocol,
)


@dataclass
class ExplainAnomalyRequest:
    """Request for anomaly explanation."""

    detector_id: UUID
    dataset: Dataset
    anomaly_indices: list[int] | None = None  # None = explain all anomalies
    explanation_method: str = "shap"  # shap, lime, feature_importance
    n_samples: int = 100  # For sampling-based methods


@dataclass
class ExplainAnomalyResponse:
    """Response with anomaly explanations."""

    explanations: dict[int, AnomalyExplanation]
    global_importance: dict[str, float] | None = None
    method_used: str = "shap"


@dataclass
class AnomalyExplanation:
    """Explanation for a single anomaly."""

    index: int
    score: float
    feature_contributions: dict[str, float]
    top_features: list[tuple[str, float]]
    explanation_text: str | None = None


class ExplainAnomalyUseCase:
    """Use case for explaining why points are anomalous."""

    def __init__(self, detector_repository: DetectorRepositoryProtocol):
        """Initialize the use case.

        Args:
            detector_repository: Repository for detectors
        """
        self.detector_repository = detector_repository

    async def execute(self, request: ExplainAnomalyRequest) -> ExplainAnomalyResponse:
        """Execute anomaly explanation.

        Args:
            request: Explanation request

        Returns:
            Explanation response

        Raises:
            ValueError: If detector doesn't support explanations
        """
        # Load detector
        detector = self.detector_repository.find_by_id(request.detector_id)
        if detector is None:
            raise ValueError(f"Detector {request.detector_id} not found")

        # Check if detector supports explanations
        if not isinstance(detector, ExplainableDetectorProtocol):
            # Fallback to basic feature analysis
            return await self._basic_explanation(detector, request)

        # Get anomaly indices if not provided
        if request.anomaly_indices is None:
            result = detector.detect(request.dataset)
            anomaly_indices = result.anomaly_indices.tolist()
        else:
            anomaly_indices = request.anomaly_indices

        # Get explanations based on method
        if request.explanation_method == "shap":
            return await self._shap_explanation(
                detector, request.dataset, anomaly_indices, request.n_samples
            )
        elif request.explanation_method == "lime":
            return await self._lime_explanation(
                detector, request.dataset, anomaly_indices, request.n_samples
            )
        elif request.explanation_method == "feature_importance":
            return await self._feature_importance_explanation(
                detector, request.dataset, anomaly_indices
            )
        else:
            raise ValueError(
                f"Unknown explanation method: {request.explanation_method}"
            )

    async def _shap_explanation(
        self,
        detector: ExplainableDetectorProtocol,
        dataset: Dataset,
        indices: list[int],
        n_samples: int,
    ) -> ExplainAnomalyResponse:
        """Generate SHAP-based explanations."""
        try:
            import shap
        except ImportError:
            # Fallback if SHAP not available
            return await self._basic_explanation(detector, dataset, indices)

        # Get detector's score function
        def score_function(X: np.ndarray) -> np.ndarray:
            temp_dataset = Dataset(
                name="temp", data=pd.DataFrame(X, columns=dataset.feature_names)
            )
            scores = detector.score(temp_dataset)
            return np.array([s.value for s in scores])

        # Create SHAP explainer
        background = dataset.features.sample(n=min(n_samples, len(dataset.features)))
        explainer = shap.KernelExplainer(score_function, background.values)

        # Get explanations for anomalies
        anomaly_data = dataset.features.iloc[indices]
        shap_values = explainer.shap_values(anomaly_data.values)

        # Convert to our format
        explanations = {}
        for i, idx in enumerate(indices):
            feature_contributions = dict(
                zip(dataset.feature_names or [], shap_values[i], strict=False)
            )

            # Sort by absolute contribution
            sorted_features = sorted(
                feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True
            )

            explanations[idx] = AnomalyExplanation(
                index=idx,
                score=float(score_function(anomaly_data.iloc[i : i + 1].values)[0]),
                feature_contributions=feature_contributions,
                top_features=sorted_features[:5],
                explanation_text=self._generate_explanation_text(sorted_features[:3]),
            )

        # Global importance
        global_importance = dict(
            zip(
                dataset.feature_names or [],
                np.abs(shap_values).mean(axis=0),
                strict=False,
            )
        )

        return ExplainAnomalyResponse(
            explanations=explanations,
            global_importance=global_importance,
            method_used="shap",
        )

    async def _lime_explanation(
        self,
        detector: ExplainableDetectorProtocol,
        dataset: Dataset,
        indices: list[int],
        n_samples: int,
    ) -> ExplainAnomalyResponse:
        """Generate LIME-based explanations."""
        try:
            from lime.lime_tabular import LimeTabularExplainer
        except ImportError:
            # Fallback if LIME not available
            return await self._basic_explanation(detector, dataset, indices)

        # Create LIME explainer
        explainer = LimeTabularExplainer(
            dataset.features.values,
            feature_names=dataset.feature_names,
            mode="regression",
            discretize_continuous=True,
        )

        # Score function for LIME
        def score_function(X: np.ndarray) -> np.ndarray:
            temp_dataset = Dataset(
                name="temp", data=pd.DataFrame(X, columns=dataset.feature_names)
            )
            scores = detector.score(temp_dataset)
            return np.array([s.value for s in scores])

        # Get explanations
        explanations = {}
        for idx in indices:
            instance = dataset.features.iloc[idx].values

            # Get LIME explanation
            exp = explainer.explain_instance(
                instance,
                score_function,
                num_features=len(dataset.feature_names or []),
                num_samples=n_samples,
            )

            # Convert to our format
            feature_contributions = dict(exp.as_list())
            sorted_features = sorted(
                feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True
            )

            explanations[idx] = AnomalyExplanation(
                index=idx,
                score=float(score_function(instance.reshape(1, -1))[0]),
                feature_contributions=feature_contributions,
                top_features=sorted_features[:5],
                explanation_text=self._generate_explanation_text(sorted_features[:3]),
            )

        return ExplainAnomalyResponse(explanations=explanations, method_used="lime")

    async def _feature_importance_explanation(
        self,
        detector: ExplainableDetectorProtocol,
        dataset: Dataset,
        indices: list[int],
    ) -> ExplainAnomalyResponse:
        """Use detector's feature importance if available."""
        # Get global feature importance
        global_importance = detector.feature_importances()

        # Get per-anomaly explanations
        explanations_dict = detector.explain(dataset, indices)

        explanations = {}
        for idx in indices:
            if idx in explanations_dict:
                exp_data = explanations_dict[idx]
                feature_contributions = exp_data.get("feature_contributions", {})

                sorted_features = sorted(
                    feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True
                )

                explanations[idx] = AnomalyExplanation(
                    index=idx,
                    score=exp_data.get("score", 0.0),
                    feature_contributions=feature_contributions,
                    top_features=sorted_features[:5],
                    explanation_text=exp_data.get("explanation"),
                )

        return ExplainAnomalyResponse(
            explanations=explanations,
            global_importance=global_importance,
            method_used="feature_importance",
        )

    async def _basic_explanation(
        self, detector: Any, dataset: Dataset, indices: list[int] | None = None
    ) -> ExplainAnomalyResponse:
        """Basic explanation using statistical analysis."""
        # Get detection results
        result = detector.detect(dataset)

        if indices is None:
            indices = result.anomaly_indices.tolist()

        # Calculate feature statistics
        normal_data = dataset.features.iloc[result.normal_indices]

        explanations = {}
        for idx in indices:
            anomaly_point = dataset.features.iloc[idx]

            # Calculate z-scores relative to normal data
            z_scores = {}
            feature_contributions = {}

            for col in dataset.get_numeric_features():
                if col == dataset.target_column:
                    continue

                mean = normal_data[col].mean()
                std = normal_data[col].std()

                if std > 0:
                    z_score = abs((anomaly_point[col] - mean) / std)
                    z_scores[col] = z_score
                    # Simple contribution based on deviation
                    feature_contributions[col] = float(z_score / 3.0)  # Normalize
                else:
                    z_scores[col] = 0.0
                    feature_contributions[col] = 0.0

            # Sort by z-score
            sorted_features = sorted(z_scores.items(), key=lambda x: x[1], reverse=True)

            explanations[idx] = AnomalyExplanation(
                index=idx,
                score=result.scores[idx].value,
                feature_contributions=feature_contributions,
                top_features=sorted_features[:5],
                explanation_text=self._generate_statistical_explanation(
                    sorted_features[:3], anomaly_point, normal_data
                ),
            )

        return ExplainAnomalyResponse(
            explanations=explanations, method_used="statistical"
        )

    def _generate_explanation_text(self, top_features: list[tuple[str, float]]) -> str:
        """Generate human-readable explanation."""
        if not top_features:
            return "No significant features identified"

        parts = []
        for feature, contribution in top_features[:3]:
            direction = "increases" if contribution > 0 else "decreases"
            parts.append(
                f"{feature} {direction} anomaly score by {abs(contribution):.2f}"
            )

        return "Anomalous because: " + "; ".join(parts)

    def _generate_statistical_explanation(
        self,
        top_features: list[tuple[str, float]],
        anomaly_point: pd.Series,
        normal_data: pd.DataFrame,
    ) -> str:
        """Generate explanation based on statistical deviation."""
        if not top_features:
            return "No significant deviations found"

        parts = []
        for feature, z_score in top_features[:3]:
            value = anomaly_point[feature]
            normal_mean = normal_data[feature].mean()

            if value > normal_mean:
                direction = "above"
            else:
                direction = "below"

            parts.append(
                f"{feature} is {z_score:.1f} std deviations {direction} normal"
            )

        return "Anomalous because: " + "; ".join(parts)
