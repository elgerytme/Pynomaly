"""
ModelSelector Domain Service

Provides intelligent model selection capabilities for machine learning models
including multi-objective optimization, Pareto front analysis, and statistical
significance testing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import ttest_ind

# Use local entities and create protocols for missing services
from ..entities.model_performance import ModelPerformanceMetrics
from typing import Protocol, Dict, Any

class MetricsCalculatorProtocol(Protocol):
    """Protocol for metrics calculation service."""
    def calculate_metrics(self, predictions: Any, ground_truth: Any) -> Dict[str, float]:
        """Calculate performance metrics."""
        ...

# Create a simple local implementation for now
class MetricsCalculator:
    """Basic metrics calculator implementation."""
    
    def calculate_metrics(self, predictions: Any, ground_truth: Any) -> Dict[str, float]:
        """Calculate basic performance metrics."""
        # Simple accuracy calculation for now
        if hasattr(predictions, '__len__') and hasattr(ground_truth, '__len__'):
            correct = sum(p == g for p, g in zip(predictions, ground_truth))
            accuracy = correct / len(predictions) if len(predictions) > 0 else 0.0
            return {"accuracy": accuracy}
        return {"accuracy": 0.0}


class ParetoOptimizer:
    """Multi-objective Pareto front optimizer for model selection."""

    def __init__(self, objectives: list[dict[str, Any]]):
        """Initialize Pareto optimizer.

        Args:
            objectives: List of objectives with name and direction
        """
        self.objectives = objectives

    def find_pareto_optimal(
        self, models: list[ModelPerformanceMetrics]
    ) -> list[dict[str, Any]]:
        """Find Pareto-optimal models considering multiple objectives.

        Args:
            models: List of model performance metrics

        Returns:
            List of Pareto-optimal models
        """
        if not models:
            return []

        def dominates(
            model1: ModelPerformanceMetrics, model2: ModelPerformanceMetrics
        ) -> bool:
            """Check if model1 dominates model2."""
            better_in_all = True
            strictly_better_in_one = False

            for objective in self.objectives:
                obj_name = objective["name"]
                direction = objective.get("direction", "maximize")

                # Get metric values
                val1 = self._get_metric_value(model1, obj_name)
                val2 = self._get_metric_value(model2, obj_name)

                if direction == "maximize":
                    if val1 < val2:
                        better_in_all = False
                        break
                    elif val1 > val2:
                        strictly_better_in_one = True
                else:  # minimize
                    if val1 > val2:
                        better_in_all = False
                        break
                    elif val1 < val2:
                        strictly_better_in_one = True

            return better_in_all and strictly_better_in_one

        pareto_optimal = []

        for model in models:
            is_dominated = False

            for other_model in models:
                if dominates(other_model, model):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_optimal.append({"model_id": model.model_id})

        return pareto_optimal

    def _get_metric_value(
        self, model: ModelPerformanceMetrics, metric_name: str
    ) -> float:
        """Get metric value from model performance metrics."""
        if hasattr(model, "metrics") and hasattr(model.metrics, metric_name):
            return getattr(model.metrics, metric_name)
        elif hasattr(model, "metrics") and isinstance(model.metrics, dict):
            return model.metrics.get(metric_name, 0.0)
        else:
            return 0.0


class ModelSelector:
    """Domain service for intelligent model selection."""

    def __init__(
        self, primary_metric: str = "f1_score", secondary_metrics: list[str] = None
    ):
        """Initialize ModelSelector.

        Args:
            primary_metric: Primary metric for model ranking
            secondary_metrics: Additional metrics for multi-objective optimization
        """
        self.primary_metric = primary_metric
        self.secondary_metrics = secondary_metrics or []

        # Create objectives for Pareto optimization
        objectives = [{"name": primary_metric, "direction": "maximize"}]
        for metric in self.secondary_metrics:
            objectives.append({"name": metric, "direction": "maximize"})

        self.pareto_optimizer = ParetoOptimizer(objectives)

    def rank_models(
        self, models: list[ModelPerformanceMetrics]
    ) -> list[dict[str, Any]]:
        """Rank models using Pareto front filtering and primary metric.

        Args:
            models: List of model performance metrics

        Returns:
            List of ranked models
        """
        if not models:
            return []

        # Convert models to format expected by MetricsCalculator
        model_results = {}
        for model in models:
            model_id = str(model.model_id)
            model_results[model_id] = self._convert_model_to_results(model)

        # Get rankings from MetricsCalculator
        comparison = MetricsCalculator.compare_models(
            model_results, primary_metric=self.primary_metric
        )

        # Get Pareto front
        pareto_optimal = self.pareto_optimizer.find_pareto_optimal(models)
        pareto_model_ids = {str(item["model_id"]) for item in pareto_optimal}

        # Filter rankings to only include Pareto-optimal models
        primary_rankings = comparison["rankings"].get(self.primary_metric, [])
        filtered_rankings = [
            rank for rank in primary_rankings if rank["model"] in pareto_model_ids
        ]

        return filtered_rankings

    def significant_difference(
        self,
        model1: ModelPerformanceMetrics,
        model2: ModelPerformanceMetrics,
        alpha: float = 0.05,
    ) -> bool:
        """Test for statistical significance between two models.

        Args:
            model1: First model performance metrics
            model2: Second model performance metrics
            alpha: Significance level

        Returns:
            True if there's a significant difference
        """
        # Get primary metric values
        val1 = self._get_metric_value(model1, self.primary_metric)
        val2 = self._get_metric_value(model2, self.primary_metric)

        # Check if metrics are available
        if val1 == 0.0 and val2 == 0.0:
            # If both are 0, check if the metric actually exists
            if not self._has_metric(
                model1, self.primary_metric
            ) or not self._has_metric(model2, self.primary_metric):
                raise KeyError(
                    f"Primary metric '{self.primary_metric}' not found in model metrics"
                )

        # Check if models have cross-validation results
        cv_scores1 = getattr(model1, "cv_scores", {}).get(self.primary_metric, None)
        cv_scores2 = getattr(model2, "cv_scores", {}).get(self.primary_metric, None)

        if cv_scores1 is not None and cv_scores2 is not None:
            # Use actual cross-validation scores for statistical testing
            array1 = np.array(cv_scores1)
            array2 = np.array(cv_scores2)

            # Ensure we have enough samples for meaningful test
            if len(array1) >= 3 and len(array2) >= 3:
                # Perform t-test with actual CV scores
                _, p_value = ttest_ind(array1, array2)
                return bool(p_value < alpha)

        # Fallback: use effect size based comparison for single point estimates
        effect_size = abs(val1 - val2) / max(
            val1, val2, 0.001
        )  # Avoid division by zero

        # Consider significant if effect size > 5% and absolute difference > threshold
        threshold = (
            0.02 if self.primary_metric in ["precision", "recall", "f1"] else 0.05
        )

        return effect_size > 0.05 and abs(val1 - val2) > threshold

    def select_best_model(
        self, models: list[ModelPerformanceMetrics]
    ) -> dict[str, Any]:
        """Select the best model with detailed rationale.

        Args:
            models: List of model performance metrics

        Returns:
            Dictionary with selected model and rationale
        """
        if not models:
            return {"decision": "No suitable models found", "rationale": []}

        # Rank models
        ranked_models = self.rank_models(models)

        if not ranked_models:
            return {"decision": "No suitable models found", "rationale": []}

        # Select best model
        best_model = ranked_models[0]
        selected_model = best_model["model"]

        # Build rationale
        rationale = []
        rationale.append(
            f"Selected {selected_model} based on primary ranking of {self.primary_metric}"
        )
        rationale.append(f"{selected_model} is part of the Pareto front")

        # Check significance with other models
        best_model_metrics = next(
            (m for m in models if str(m.model_id) == selected_model), None
        )

        if best_model_metrics and len(models) > 1:
            other_models = [m for m in models if str(m.model_id) != selected_model]
            significant_differences = []

            for other_model in other_models:
                if self.significant_difference(best_model_metrics, other_model):
                    significant_differences.append(str(other_model.model_id))

            if significant_differences:
                rationale.append(
                    f"Significant performance difference from models: {', '.join(significant_differences)}"
                )
            else:
                rationale.append(
                    "No significant difference detected, selection based on lack of significant difference"
                )

        return {"selected_model": selected_model, "rationale": rationale}

    def _convert_model_to_results(
        self, model: ModelPerformanceMetrics
    ) -> dict[str, Any]:
        """Convert ModelPerformanceMetrics to format expected by MetricsCalculator."""
        results = {}

        # Get metrics from the model
        if hasattr(model, "metrics"):
            if hasattr(model.metrics, "to_dict"):
                # PerformanceMetrics value object
                metrics_dict = model.metrics.to_dict()
                for key, value in metrics_dict.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        results[key] = {"value": value}
            elif isinstance(model.metrics, dict):
                # Dictionary of metrics
                for key, value in model.metrics.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        results[key] = {"value": value}

        return results

    def _get_metric_value(
        self, model: ModelPerformanceMetrics, metric_name: str
    ) -> float:
        """Get metric value from model performance metrics."""
        if hasattr(model, "metrics"):
            if hasattr(model.metrics, metric_name):
                return getattr(model.metrics, metric_name)
            elif isinstance(model.metrics, dict):
                return model.metrics.get(metric_name, 0.0)

        return 0.0

    def _has_metric(self, model: ModelPerformanceMetrics, metric_name: str) -> bool:
        """Check if model has the specified metric."""
        if hasattr(model, "metrics"):
            if hasattr(model.metrics, metric_name):
                return True
            elif isinstance(model.metrics, dict):
                return metric_name in model.metrics

        return False
