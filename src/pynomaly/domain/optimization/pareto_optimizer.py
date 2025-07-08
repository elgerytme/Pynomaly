"""
Pareto Optimizer for Multi-Objective Model Selection

This module provides Pareto front optimization for selecting models
based on multiple conflicting objectives.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class Objective:
    """Represents an optimization objective."""
    name: str
    direction: str  # 'max' or 'min'
    weight: float = 1.0


class ParetoOptimizer:
    """
    Pareto front optimizer for multi-objective model selection.
    
    This class finds the Pareto-optimal set of models considering
    multiple conflicting objectives (e.g., accuracy vs. speed).
    """
    
    def __init__(self, objectives: List[Dict[str, Any]]):
        """
        Initialize ParetoOptimizer.
        
        Args:
            objectives: List of objective dictionaries with 'name', 'direction', and optional 'weight'
        """
        self.objectives = [
            Objective(
                name=obj['name'],
                direction=obj['direction'],
                weight=obj.get('weight', 1.0)
            )
            for obj in objectives
        ]
    
    def find_pareto_optimal(self, models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find Pareto-optimal models from a list of models.
        
        Args:
            models: List of model dictionaries with metrics
            
        Returns:
            List of Pareto-optimal models
        """
        if not models:
            return []
        
        # Extract objective values for each model
        objective_values = []
        for model in models:
            values = {}
            for objective in self.objectives:
                if objective.name in model.get('metrics', {}):
                    value = model['metrics'][objective.name]
                    # Handle nested metric structure
                    if isinstance(value, dict) and 'value' in value:
                        value = value['value']
                    values[objective.name] = value
                else:
                    # If objective not found, use default value
                    values[objective.name] = 0.0
            objective_values.append(values)
        
        # Find Pareto front
        pareto_indices = self._find_pareto_front(objective_values)
        
        # Return Pareto-optimal models
        return [models[i] for i in pareto_indices]
    
    def _find_pareto_front(self, objective_values: List[Dict[str, float]]) -> List[int]:
        """
        Find indices of Pareto-optimal solutions.
        
        Args:
            objective_values: List of objective value dictionaries
            
        Returns:
            List of indices of Pareto-optimal solutions
        """
        n_models = len(objective_values)
        pareto_indices = []
        
        for i in range(n_models):
            is_dominated = False
            
            for j in range(n_models):
                if i != j and self._dominates(objective_values[j], objective_values[i]):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_indices.append(i)
        
        return pareto_indices
    
    def _dominates(self, solution1: Dict[str, float], solution2: Dict[str, float]) -> bool:
        """
        Check if solution1 dominates solution2.
        
        Args:
            solution1: First solution's objective values
            solution2: Second solution's objective values
            
        Returns:
            True if solution1 dominates solution2
        """
        better_in_all = True
        strictly_better_in_one = False
        
        for objective in self.objectives:
            obj_name = objective.name
            val1 = solution1.get(obj_name, 0.0)
            val2 = solution2.get(obj_name, 0.0)
            
            # Apply direction (maximize or minimize)
            if objective.direction == 'min':
                val1 = -val1
                val2 = -val2
            
            if val1 < val2:
                better_in_all = False
                break
            elif val1 > val2:
                strictly_better_in_one = True
        
        return better_in_all and strictly_better_in_one
    
    def compute_hypervolume(self, models: List[Dict[str, Any]], reference_point: Optional[Dict[str, float]] = None) -> float:
        """
        Compute hypervolume indicator for the Pareto front.
        
        Args:
            models: List of models
            reference_point: Reference point for hypervolume calculation
            
        Returns:
            Hypervolume value
        """
        pareto_models = self.find_pareto_optimal(models)
        
        if not pareto_models:
            return 0.0
        
        # Extract objective values
        objective_values = []
        for model in pareto_models:
            values = []
            for objective in self.objectives:
                if objective.name in model.get('metrics', {}):
                    value = model['metrics'][objective.name]
                    if isinstance(value, dict) and 'value' in value:
                        value = value['value']
                    # Apply direction
                    if objective.direction == 'min':
                        value = -value
                    values.append(value)
                else:
                    values.append(0.0)
            objective_values.append(values)
        
        # Compute hypervolume (simplified 2D case)
        if len(self.objectives) == 2:
            return self._compute_hypervolume_2d(objective_values, reference_point)
        else:
            # For higher dimensions, use approximation
            return self._compute_hypervolume_approximation(objective_values, reference_point)
    
    def _compute_hypervolume_2d(self, points: List[List[float]], reference_point: Optional[Dict[str, float]] = None) -> float:
        """Compute hypervolume for 2D case."""
        if not points:
            return 0.0
        
        # Sort points by first objective
        sorted_points = sorted(points, key=lambda p: p[0], reverse=True)
        
        # Use default reference point if not provided
        if reference_point is None:
            ref_x = min(p[0] for p in sorted_points) - 1
            ref_y = min(p[1] for p in sorted_points) - 1
        else:
            ref_x = reference_point.get(self.objectives[0].name, 0.0)
            ref_y = reference_point.get(self.objectives[1].name, 0.0)
            if self.objectives[0].direction == 'min':
                ref_x = -ref_x
            if self.objectives[1].direction == 'min':
                ref_y = -ref_y
        
        # Compute hypervolume
        hypervolume = 0.0
        prev_y = ref_y
        
        for point in sorted_points:
            x, y = point
            if y > prev_y:
                hypervolume += (x - ref_x) * (y - prev_y)
                prev_y = y
        
        return hypervolume
    
    def _compute_hypervolume_approximation(self, points: List[List[float]], reference_point: Optional[Dict[str, float]] = None) -> float:
        """Compute hypervolume approximation for higher dimensions."""
        if not points:
            return 0.0
        
        # Simple approximation: sum of normalized objective values
        total_volume = 0.0
        
        for point in points:
            volume = 1.0
            for i, value in enumerate(point):
                # Normalize value (assuming values are between 0 and 1)
                normalized_value = max(0.0, min(1.0, value))
                volume *= normalized_value
            total_volume += volume
        
        return total_volume / len(points)
    
    def get_optimization_summary(self, models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get comprehensive optimization summary.
        
        Args:
            models: List of models to analyze
            
        Returns:
            Dictionary with optimization analysis
        """
        pareto_models = self.find_pareto_optimal(models)
        
        summary = {
            'total_models': len(models),
            'pareto_optimal_models': len(pareto_models),
            'pareto_efficiency': len(pareto_models) / len(models) if models else 0.0,
            'hypervolume': self.compute_hypervolume(models),
            'objectives': [
                {
                    'name': obj.name,
                    'direction': obj.direction,
                    'weight': obj.weight
                }
                for obj in self.objectives
            ],
            'pareto_front': pareto_models
        }
        
        # Add objective statistics
        if models:
            objective_stats = {}
            for objective in self.objectives:
                values = []
                for model in models:
                    if objective.name in model.get('metrics', {}):
                        value = model['metrics'][objective.name]
                        if isinstance(value, dict) and 'value' in value:
                            value = value['value']
                        values.append(value)
                
                if values:
                    objective_stats[objective.name] = {
                        'min': min(values),
                        'max': max(values),
                        'mean': np.mean(values),
                        'std': np.std(values)
                    }
            
            summary['objective_statistics'] = objective_stats
        
        return summary
