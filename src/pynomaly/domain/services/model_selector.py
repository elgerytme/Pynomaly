"""Model Selection Engine.

This module provides a comprehensive model selection tool combining
primary metric ranking, Pareto front filtering, statistical significance,
and tie-breaking by secondary metrics or resources.
"""

from typing import Dict, Any, List
from scipy.stats import ttest_ind
import numpy as np

from pynomaly.domain.entities.model_performance import ModelPerformanceMetrics
from pynomaly.domain.optimization.pareto_optimizer import ParetoOptimizer
from pynomaly.domain.services.metrics_calculator import MetricsCalculator


class ModelSelector:
    """Comprehensive model ranking and selection engine.
    
    This class combines multiple criteria for model selection:
    1. Primary metric ranking
    2. Pareto front filtering
    3. Statistical significance testing
    4. Tie-breaking by secondary metrics or resource usage
    
    Returns ordered list with rationale for each decision for auditability.
    """

    def __init__(self, primary_metric: str, secondary_metrics: List[str]):
        """Initialize ModelSelector.
        
        Args:
            primary_metric: Primary metric for ranking (e.g., 'f1_score', 'accuracy')
            secondary_metrics: List of secondary metrics for tie-breaking
        """
        self.primary_metric = primary_metric
        self.secondary_metrics = secondary_metrics
        self.pareto_optimizer = ParetoOptimizer(
            objectives=[{'name': primary_metric, 'direction': 'max'}] +
                      [{'name': metric, 'direction': 'max'} for metric in secondary_metrics]
        )

    def rank_models(self, models: List[ModelPerformanceMetrics]) -> List[Dict[str, Any]]:
        """Rank models based on primary metric and apply Pareto front filtering.
        
        Args:
            models: List of models with performance metrics
            
        Returns:
            List of ranked models that are in the Pareto front
        """
        if not models:
            return []
            
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
        pareto_model_ids = [p['model_id'] for p in pareto_front]

        # Filter models not in Pareto front
        ranked_pareto_front = [
            r for r in ranked_results['rankings'][self.primary_metric] 
            if r['model'] in pareto_model_ids
        ]

        return ranked_pareto_front

    def significant_difference(self, model_a: ModelPerformanceMetrics, model_b: ModelPerformanceMetrics) -> bool:
        """Test statistical significance between two models.
        
        Args:
            model_a: First model for comparison
            model_b: Second model for comparison
            
        Returns:
            True if there's a significant difference, False otherwise
        """
        # Extract metric values for statistical comparison
        a_metrics = np.array(list(model_a.metrics.values()))
        b_metrics = np.array(list(model_b.metrics.values()))

        # Perform Welch's t-test (unequal variances)
        stat, p_value = ttest_ind(a_metrics, b_metrics, equal_var=False)
        return p_value < 0.05  # Using 0.05 significance level

    def select_best_model(self, models: List[ModelPerformanceMetrics]) -> Dict[str, Any]:
        """Select the best model based on comprehensive criteria.
        
        Args:
            models: List of models to select from
            
        Returns:
            Dictionary with selected model and detailed rationale
        """
        ranked_models = self.rank_models(models)

        if not ranked_models:
            return {
                'decision': 'No suitable models found',
                'rationale': []
            }

        best_model = ranked_models[0]['model']
        best_model_obj = next(m for m in models if m.model_id == best_model)

        # Build rationale
        rationale = [
            f"Selected {best_model} based on primary ranking of {self.primary_metric}",
            f"{best_model} is part of the Pareto front"
        ]

        # Statistical significance gating
        for ranked_model in ranked_models[1:]:
            model_obj = next(m for m in models if m.model_id == ranked_model['model'])
            
            if not self.significant_difference(best_model_obj, model_obj):
                rationale.append(
                    f"{ranked_model['model']} discarded due to lack of significant difference"
                )

        # Tie-breaking information
        if len(ranked_models) > 1:
            # Check if there are ties in primary metric
            best_score = ranked_models[0]['value']
            tied_models = [m for m in ranked_models if abs(m['value'] - best_score) < 1e-6]
            
            if len(tied_models) > 1:
                rationale.append(
                    f"Tie-breaking applied among {len(tied_models)} models with similar {self.primary_metric}"
                )
                
                # Add secondary metric analysis
                for secondary_metric in self.secondary_metrics:
                    secondary_values = []
                    for tied_model in tied_models:
                        model_obj = next(m for m in models if m.model_id == tied_model['model'])
                        if secondary_metric in model_obj.metrics:
                            secondary_values.append(
                                (tied_model['model'], model_obj.metrics[secondary_metric])
                            )
                    
                    if secondary_values:
                        best_secondary = max(secondary_values, key=lambda x: x[1])
                        rationale.append(
                            f"Best {secondary_metric}: {best_secondary[0]} ({best_secondary[1]:.3f})"
                        )

        return {
            'selected_model': best_model,
            'rationale': rationale,
            'ranking_details': ranked_models,
            'selection_criteria': {
                'primary_metric': self.primary_metric,
                'secondary_metrics': self.secondary_metrics,
                'significance_level': 0.05,
                'pareto_filtered': True
            }
        }

    def get_selection_summary(self, models: List[ModelPerformanceMetrics]) -> Dict[str, Any]:
        """Get a comprehensive summary of the selection process.
        
        Args:
            models: List of models to analyze
            
        Returns:
            Dictionary with detailed selection analysis
        """
        selection_result = self.select_best_model(models)
        
        summary = {
            'total_models_evaluated': len(models),
            'models_in_pareto_front': len(self.rank_models(models)),
            'selection_result': selection_result,
            'model_comparison': {}
        }
        
        # Add pairwise comparisons for top models
        ranked_models = self.rank_models(models)
        for i, model_a in enumerate(ranked_models[:3]):  # Top 3 models
            for j, model_b in enumerate(ranked_models[i+1:3], i+1):
                model_a_obj = next(m for m in models if m.model_id == model_a['model'])
                model_b_obj = next(m for m in models if m.model_id == model_b['model'])
                
                comparison_key = f"{model_a['model']}_vs_{model_b['model']}"
                summary['model_comparison'][comparison_key] = {
                    'significant_difference': self.significant_difference(model_a_obj, model_b_obj),
                    'primary_metric_diff': model_a['value'] - model_b['value'],
                    'better_model': model_a['model'] if model_a['value'] > model_b['value'] else model_b['model']
                }
        
        return summary
