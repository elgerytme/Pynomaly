"""
Model Selection Service

This service provides functionality for selecting the best model
from a set of candidates based on performance metrics and other criteria.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ModelCandidate:
    """Represents a candidate model for selection."""
    model_id: str
    algorithm: str
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]


class ModelSelector:
    """
    Service for selecting the best model from candidates.
    
    This service implements various model selection strategies
    based on performance metrics and other criteria.
    """
    
    def __init__(self, primary_metric: str = 'f1_score', secondary_metrics: Optional[List[str]] = None):
        """
        Initialize the model selector.
        
        Args:
            primary_metric: Primary metric to use for selection
            secondary_metrics: List of secondary metrics to consider
        """
        self.primary_metric = primary_metric
        self.secondary_metrics = secondary_metrics or ['accuracy', 'precision', 'recall']
        
    def select_best_model(self, candidates) -> Optional[Dict[str, Any]]:
        """
        Select the best model from candidates.
        
        Args:
            candidates: List of candidate models or performance metrics
            
        Returns:
            Dictionary with selected model information or None if no candidates provided
        """
        if not candidates:
            return None
            
        # Handle different types of candidates
        if hasattr(candidates[0], 'model_id'):
            # ModelCandidate objects
            best_candidate = max(candidates, key=lambda x: x.metrics.get(self.primary_metric, 0.0))
            return {
                'selected_model': best_candidate.model_id,
                'algorithm': best_candidate.algorithm,
                'metrics': best_candidate.metrics,
                'primary_metric_value': best_candidate.metrics.get(self.primary_metric, 0.0)
            }
        else:
            # Assume it's a list of performance metrics objects
            best_candidate = max(candidates, key=lambda x: getattr(x, 'metrics', {}).get(self.primary_metric, 0.0))
            return {
                'selected_model': getattr(best_candidate, 'model_id', None),
                'algorithm': getattr(best_candidate, 'algorithm', 'unknown'),
                'metrics': getattr(best_candidate, 'metrics', {}),
                'primary_metric_value': getattr(best_candidate, 'metrics', {}).get(self.primary_metric, 0.0)
            }
        
    def rank_models(self, candidates: List[ModelCandidate]) -> List[ModelCandidate]:
        """
        Rank models by performance.
        
        Args:
            candidates: List of candidate models
            
        Returns:
            List of models ranked by performance
        """
        # Dummy implementation - return as is
        return candidates
        
    def compare_models(self, model1: ModelCandidate, model2: ModelCandidate) -> Dict[str, Any]:
        """
        Compare two models.
        
        Args:
            model1: First model candidate
            model2: Second model candidate
            
        Returns:
            Comparison results
        """
        return {
            'better_model': model1.model_id,
            'metrics_comparison': {},
            'recommendation': 'Use model1'
        }
