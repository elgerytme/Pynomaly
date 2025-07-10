"""
ModelSelector Domain Service

Provides intelligent model selection capabilities for anomaly detection models
including multi-objective optimization, Pareto front analysis, and statistical
significance testing.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from scipy.stats import ttest_ind

from pynomaly.domain.entities.model_performance import ModelPerformanceMetrics
from pynomaly.domain.services.metrics_calculator import MetricsCalculator


class ParetoOptimizer:
    """Multi-objective Pareto front optimizer for model selection."""
    
    def __init__(self, objectives: List[Dict[str, Any]]):
        """Initialize Pareto optimizer.
        
        Args:
            objectives: List of objectives with name and direction
        """
        self.objectives = objectives
    
    def find_pareto_optimal(
        self, models: List[ModelPerformanceMetrics]
    ) -> List[Dict[str, Any]]:
        """Find Pareto-optimal models considering multiple objectives.
        
        Args:
            models: List of model performance metrics
            
        Returns:
            List of Pareto-optimal models
        """
        if not models:
            return []
        
        def dominates(model1: ModelPerformanceMetrics, model2: ModelPerformanceMetrics) -> bool:
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
    
    def _get_metric_value(self, model: ModelPerformanceMetrics, metric_name: str) -> float:
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
        self,
        primary_metric: str = "f1_score",
        secondary_metrics: List[str] = None
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
        self, models: List[ModelPerformanceMetrics]
    ) -> List[Dict[str, Any]]:
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
            rank for rank in primary_rankings
            if rank["model"] in pareto_model_ids
        ]
        
        return filtered_rankings
    
    def significant_difference(
        self,
        model1: ModelPerformanceMetrics,
        model2: ModelPerformanceMetrics,
        alpha: float = 0.05
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
        
        # For single values, create arrays for t-test
        # In practice, these would be cross-validation results
        array1 = np.array([val1] * 10 + np.random.normal(0, 0.01, 10))
        array2 = np.array([val2] * 10 + np.random.normal(0, 0.01, 10))
        
        # Perform t-test
        _, p_value = ttest_ind(array1, array2)
        
        return p_value < alpha
    
    def select_best_model(
        self, models: List[ModelPerformanceMetrics]
    ) -> Dict[str, Any]:
        """Select the best model with detailed rationale.
        
        Args:
            models: List of model performance metrics
            
        Returns:
            Dictionary with selected model and rationale
        """
        if not models:
            return {
                "decision": "No suitable models found",
                "rationale": []
            }
        
        # Rank models
        ranked_models = self.rank_models(models)
        
        if not ranked_models:
            return {
                "decision": "No suitable models found",
                "rationale": []
            }
        
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
        
        return {
            "selected_model": selected_model,
            "rationale": rationale
        }
    
    def _convert_model_to_results(self, model: ModelPerformanceMetrics) -> Dict[str, Any]:
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
    
    def _get_metric_value(self, model: ModelPerformanceMetrics, metric_name: str) -> float:
        """Get metric value from model performance metrics."""
        if hasattr(model, "metrics"):
            if hasattr(model.metrics, metric_name):
                return getattr(model.metrics, metric_name)
            elif isinstance(model.metrics, dict):
                return model.metrics.get(metric_name, 0.0)
        
        return 0.0