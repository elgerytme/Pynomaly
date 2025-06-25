"""Domain service for anomaly detection explainability."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ExplanationMethod(Enum):
    """Available explanation methods."""

    SHAP = "shap"
    LIME = "lime"
    FEATURE_IMPORTANCE = "feature_importance"
    PERMUTATION_IMPORTANCE = "permutation_importance"
    PARTIAL_DEPENDENCE = "partial_dependence"
    COUNTERFACTUAL = "counterfactual"


class ExplanationType(Enum):
    """Types of explanations."""

    GLOBAL = "global"
    LOCAL = "local"
    COHORT = "cohort"


@dataclass
class FeatureContribution:
    """Represents contribution of a feature to anomaly prediction."""

    feature_name: str
    value: float
    contribution: float
    importance: float
    rank: int
    description: str | None = None


@dataclass
class LocalExplanation:
    """Local explanation for a single instance."""

    instance_id: str
    anomaly_score: float
    prediction: str
    confidence: float
    feature_contributions: list[FeatureContribution]
    explanation_method: ExplanationMethod
    model_name: str
    timestamp: str


@dataclass
class GlobalExplanation:
    """Global explanation for the entire model."""

    model_name: str
    feature_importances: dict[str, float]
    top_features: list[str]
    explanation_method: ExplanationMethod
    model_performance: dict[str, float]
    timestamp: str
    summary: str


@dataclass
class CohortExplanation:
    """Explanation for a cohort of similar instances."""

    cohort_id: str
    cohort_description: str
    instance_count: int
    common_features: list[FeatureContribution]
    explanation_method: ExplanationMethod
    model_name: str
    timestamp: str


class ExplainerProtocol(ABC):
    """Protocol for explainability implementations."""

    @abstractmethod
    def explain_local(
        self, instance: np.ndarray, model: Any, feature_names: list[str], **kwargs
    ) -> LocalExplanation:
        """Generate local explanation for a single instance."""
        pass

    @abstractmethod
    def explain_global(
        self, data: np.ndarray, model: Any, feature_names: list[str], **kwargs
    ) -> GlobalExplanation:
        """Generate global explanation for the model."""
        pass

    @abstractmethod
    def explain_cohort(
        self,
        instances: np.ndarray,
        model: Any,
        feature_names: list[str],
        cohort_id: str,
        **kwargs,
    ) -> CohortExplanation:
        """Generate explanation for a cohort of instances."""
        pass


class ExplainabilityService:
    """Domain service for generating explanations."""

    def __init__(self):
        """Initialize explainability service."""
        self._explainers: dict[ExplanationMethod, ExplainerProtocol] = {}

    def register_explainer(
        self, method: ExplanationMethod, explainer: ExplainerProtocol
    ) -> None:
        """Register an explainer for a specific method."""
        self._explainers[method] = explainer
        logger.info(f"Registered explainer for method: {method.value}")

    def get_available_methods(self) -> list[ExplanationMethod]:
        """Get list of available explanation methods."""
        return list(self._explainers.keys())

    def explain_instance(
        self,
        instance: np.ndarray | pd.Series | dict[str, Any],
        model: Any,
        feature_names: list[str],
        method: ExplanationMethod = ExplanationMethod.SHAP,
        **kwargs,
    ) -> LocalExplanation:
        """Generate explanation for a single instance."""
        if method not in self._explainers:
            raise ValueError(f"Explainer for method {method.value} not available")

        # Convert input to numpy array
        if isinstance(instance, (pd.Series, dict)):
            instance_array = self._convert_to_array(instance, feature_names)
        else:
            instance_array = np.array(instance).reshape(1, -1)

        explainer = self._explainers[method]
        return explainer.explain_local(
            instance=instance_array[0],
            model=model,
            feature_names=feature_names,
            **kwargs,
        )

    def explain_model(
        self,
        data: np.ndarray | pd.DataFrame,
        model: Any,
        feature_names: list[str],
        method: ExplanationMethod = ExplanationMethod.SHAP,
        **kwargs,
    ) -> GlobalExplanation:
        """Generate global explanation for the model."""
        if method not in self._explainers:
            raise ValueError(f"Explainer for method {method.value} not available")

        # Convert input to numpy array
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = np.array(data)

        explainer = self._explainers[method]
        return explainer.explain_global(
            data=data_array, model=model, feature_names=feature_names, **kwargs
        )

    def explain_cohort(
        self,
        instances: np.ndarray | pd.DataFrame,
        model: Any,
        feature_names: list[str],
        cohort_id: str,
        method: ExplanationMethod = ExplanationMethod.SHAP,
        **kwargs,
    ) -> CohortExplanation:
        """Generate explanation for a cohort of instances."""
        if method not in self._explainers:
            raise ValueError(f"Explainer for method {method.value} not available")

        # Convert input to numpy array
        if isinstance(instances, pd.DataFrame):
            instances_array = instances.values
        else:
            instances_array = np.array(instances)

        explainer = self._explainers[method]
        return explainer.explain_cohort(
            instances=instances_array,
            model=model,
            feature_names=feature_names,
            cohort_id=cohort_id,
            **kwargs,
        )

    def compare_explanations(
        self,
        instance: np.ndarray | pd.Series | dict[str, Any],
        model: Any,
        feature_names: list[str],
        methods: list[ExplanationMethod],
        **kwargs,
    ) -> dict[ExplanationMethod, LocalExplanation]:
        """Compare explanations from multiple methods."""
        explanations = {}

        for method in methods:
            if method in self._explainers:
                try:
                    explanation = self.explain_instance(
                        instance=instance,
                        model=model,
                        feature_names=feature_names,
                        method=method,
                        **kwargs,
                    )
                    explanations[method] = explanation
                except Exception as e:
                    logger.warning(
                        f"Failed to generate explanation with {method.value}: {e}"
                    )

        return explanations

    def _convert_to_array(
        self, instance: pd.Series | dict[str, Any], feature_names: list[str]
    ) -> np.ndarray:
        """Convert instance to numpy array."""
        if isinstance(instance, pd.Series):
            return instance.reindex(feature_names).fillna(0).values.reshape(1, -1)
        elif isinstance(instance, dict):
            return np.array([instance.get(name, 0) for name in feature_names]).reshape(
                1, -1
            )
        else:
            raise ValueError(f"Unsupported instance type: {type(instance)}")

    def get_feature_statistics(
        self, explanations: list[LocalExplanation]
    ) -> dict[str, dict[str, float]]:
        """Calculate statistics across multiple explanations."""
        feature_stats = {}

        for explanation in explanations:
            for contrib in explanation.feature_contributions:
                if contrib.feature_name not in feature_stats:
                    feature_stats[contrib.feature_name] = {
                        "contributions": [],
                        "importances": [],
                        "values": [],
                    }

                feature_stats[contrib.feature_name]["contributions"].append(
                    contrib.contribution
                )
                feature_stats[contrib.feature_name]["importances"].append(
                    contrib.importance
                )
                feature_stats[contrib.feature_name]["values"].append(contrib.value)

        # Calculate summary statistics
        summary_stats = {}
        for feature_name, stats in feature_stats.items():
            summary_stats[feature_name] = {
                "mean_contribution": np.mean(stats["contributions"]),
                "std_contribution": np.std(stats["contributions"]),
                "mean_importance": np.mean(stats["importances"]),
                "std_importance": np.std(stats["importances"]),
                "mean_value": np.mean(stats["values"]),
                "std_value": np.std(stats["values"]),
                "count": len(stats["contributions"]),
            }

        return summary_stats

    def rank_features_by_importance(
        self,
        explanations: list[LocalExplanation | GlobalExplanation],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Rank features by their average importance across explanations."""
        feature_importances = {}

        for explanation in explanations:
            if isinstance(explanation, LocalExplanation):
                for contrib in explanation.feature_contributions:
                    if contrib.feature_name not in feature_importances:
                        feature_importances[contrib.feature_name] = []
                    feature_importances[contrib.feature_name].append(contrib.importance)
            elif isinstance(explanation, GlobalExplanation):
                for feature_name, importance in explanation.feature_importances.items():
                    if feature_name not in feature_importances:
                        feature_importances[feature_name] = []
                    feature_importances[feature_name].append(importance)

        # Calculate average importance
        avg_importances = {
            feature: np.mean(importances)
            for feature, importances in feature_importances.items()
        }

        # Sort by importance and return top k
        sorted_features = sorted(
            avg_importances.items(), key=lambda x: x[1], reverse=True
        )

        return sorted_features[:top_k]
