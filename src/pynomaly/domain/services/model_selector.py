"""Model Selection Engine.

This module provides a comprehensive model selection tool combining
primary metric ranking, Pareto front filtering, statistical significance,
and tie-breaking by secondary metrics or resources.
"""

from typing import Dict, Any, List
from scipy.stats import ttest_ind
import numpy as np

from src.pynomaly.domain.entities.model_performance import ModelPerformanceMetrics
from src.pynomaly.domain.optimization.pareto_optimizer import ParetoOptimizer
from src.pynomaly.domain.services.metrics_calculator import MetricsCalculator

class ModelSelector:
    """Selector for model ranking and selection using multiple criteria."""

    def __init__(self, primary_metric: str, secondary_metrics: List[str]):
        self.primary_metric = primary_metric
        self.secondary_metrics = secondary_metrics
        self.pareto_optimizer = ParetoOptimizer(
            objectives=[{'name': primary_metric, 'direction': 'max'}] +
                      [{'name': metric, 'direction': 'max'} for metric in secondary_metrics]
        )

    def rank_models(self, models: List[ModelPerformanceMetrics]) -> List[Dict[str, Any]]:
        """Rank models based on primary metric and apply Pareto front filtering."""
        model_dicts = [{
            'model_id': model.model_id,
            'metrics': model.metrics
        } for model in models]

        # Primary Metric Ranking
        ranked_results = MetricsCalculator.compare_models(
            {m['model_id']: m['metrics'] for m in model_dicts},
            primary_metric=self.primary_metric
        )

        # Pareto Front Filtering
        pareto_front = self.pareto_optimizer.find_pareto_optimal(model_dicts)

        # Filter models not in Pareto front
        ranked_pareto_front = [r for r in ranked_results['rankings'][self.primary_metric] 
                               if r['model'] in [p['model_id'] for p in pareto_front]]

        return ranked_pareto_front

    def significant_difference(self, model_a: ModelPerformanceMetrics, model_b: ModelPerformanceMetrics) -> bool:
        """Statistical significance test between two models."""
        a_metrics = np.array(list(model_a.metrics.values()))
        b_metrics = np.array(list(model_b.metrics.values()))

        stat, p_value = ttest_ind(a_metrics, b_metrics, equal_var=False)
        return p_value < 0.05  # Using 0.05 significance level

    def select_best_model(self, models: List[ModelPerformanceMetrics]) -> Dict[str, Any]:
        """Select the best model based on the criteria and return decision rationale."""
        ranked_models = self.rank_models(models)

        if not ranked_models:
            return {'decision': 'No suitable models found', 'rationale': []}

        best_model = ranked_models[0]['model']

        rationale = [f"Selected {best_model} based on primary ranking of {self.primary_metric}" ,
                     f"{best_model} is part of the Pareto front"]

        for model in ranked_models[1:]:
            if not self.significant_difference(
                    next(m for m in models if m.model_id == best_model),
                    next(m for m in models if m.model_id == model['model'])
            ):
                rationale.append(f"{model['model']} discarded due to lack of significant difference")

        return {
            'selected_model': best_model,
            'rationale': rationale
        }
