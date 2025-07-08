"""
Pareto optimization engine for multi-objective optimization.

This module provides fast numpy-based Pareto optimization with optional
numba acceleration for performance-critical scenarios.
"""

from typing import Any, Dict, List, Optional
import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator that does nothing
    def njit(func):
        return func


class ParetoOptimizer:
    """Pareto optimization engine for multi-objective optimization.
    
    This class provides fast numpy-based Pareto optimization with optional
    numba acceleration for performance-critical scenarios.
    """
    
    def __init__(self, objectives: List[Dict[str, str]], epsilon: float = 0.0, 
                 use_numba: bool = True):
        """
        Initialize the ParetoOptimizer.

        Args:
            objectives: A list of dictionaries specifying the objectives, with 'name' and 'direction' keys.
                       E.g., [{'name': 'f1_score', 'direction': 'max'}, {'name': 'precision', 'direction': 'max'}]
            epsilon: Epsilon value for epsilon-dominance (default is 0.0, which means no epsilon-dominance).
            use_numba: Whether to use numba acceleration if available (default: True).
        """
        self.objectives = objectives
        self.epsilon = epsilon
        self.use_numba = use_numba and NUMBA_AVAILABLE
        
        # Validate objectives
        self._validate_objectives(objectives)
    
    def _validate_objectives(self, objectives: List[Dict[str, str]]) -> None:
        """Validate the objectives configuration."""
        if not objectives:
            raise ValueError("At least one objective must be specified")
        
        for obj in objectives:
            if 'name' not in obj or 'direction' not in obj:
                raise ValueError("Each objective must have 'name' and 'direction' keys")
            
            if obj['direction'] not in ['max', 'min']:
                raise ValueError(f"Direction must be 'max' or 'min', got: {obj['direction']}")

    def find_pareto_optimal(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find Pareto-optimal results considering multiple objectives.

        Args:
            results: List of results with metrics to evaluate.
            
        Returns:
            List of Pareto-optimal results
        """
        if not results:
            return []
        
        # Convert results to objective scores matrix
        scores = self._extract_scores(results)
        
        # Apply epsilon-dominance for noisy metrics
        if self.epsilon > 0:
            scores = self._apply_epsilon_dominance(scores)
        
        # Find Pareto-optimal points
        if self.use_numba:
            is_pareto = self._identify_pareto_numba(scores)
        else:
            is_pareto = self._identify_pareto_numpy(scores)
        
        return [result for idx, result in enumerate(results) if is_pareto[idx]]
    
    def _extract_scores(self, results: List[Dict[str, Any]]) -> np.ndarray:
        """Extract objective scores from results."""
        scores = []
        for result in results:
            result_scores = []
            for obj in self.objectives:
                score = result.get('metrics', {}).get(obj['name'], 0.0)
                # Convert to maximization problem (negate for minimization)
                if obj['direction'] == 'min':
                    score = -score
                result_scores.append(score)
            scores.append(result_scores)
        
        return np.array(scores, dtype=np.float64)
    
    def _apply_epsilon_dominance(self, scores: np.ndarray) -> np.ndarray:
        """Apply epsilon-dominance for handling noisy metrics."""
        # Add small random noise to handle epsilon-dominance
        noise = np.random.uniform(-self.epsilon, self.epsilon, scores.shape)
        return scores + noise
    
    @staticmethod
    @njit
    def _identify_pareto_numba(scores: np.ndarray) -> np.ndarray:
        """Identify Pareto-optimal points using numba acceleration."""
        n_points = scores.shape[0]
        is_pareto = np.ones(n_points, dtype=np.bool_)
        
        for i in range(n_points):
            for j in range(n_points):
                if i != j:
                    # Check if j dominates i
                    dominates = True
                    strictly_better = False
                    
                    for k in range(scores.shape[1]):
                        if scores[j, k] < scores[i, k]:
                            dominates = False
                            break
                        elif scores[j, k] > scores[i, k]:
                            strictly_better = True
                    
                    if dominates and strictly_better:
                        is_pareto[i] = False
                        break
        
        return is_pareto
    
    def _identify_pareto_numpy(self, scores: np.ndarray) -> np.ndarray:
        """Identify Pareto-optimal points using pure numpy."""
        n_points = scores.shape[0]
        is_pareto = np.ones(n_points, dtype=bool)
        
        for i in range(n_points):
            for j in range(n_points):
                if i != j:
                    # Check if j dominates i
                    dominates = np.all(scores[j] >= scores[i])
                    strictly_better = np.any(scores[j] > scores[i])
                    
                    if dominates and strictly_better:
                        is_pareto[i] = False
                        break
        
        return is_pareto
    
    def get_pareto_front(self, results: List[Dict[str, Any]]) -> np.ndarray:
        """
        Get the Pareto front (objective space representation).
        
        Args:
            results: List of results with metrics to evaluate.
            
        Returns:
            Array of Pareto-optimal objective values
        """
        pareto_optimal = self.find_pareto_optimal(results)
        if not pareto_optimal:
            return np.array([])
        
        return self._extract_scores(pareto_optimal)
    
    def hypervolume(self, results: List[Dict[str, Any]], reference_point: Optional[np.ndarray] = None) -> float:
        """
        Calculate the hypervolume indicator for the Pareto front.
        
        Args:
            results: List of results with metrics to evaluate.
            reference_point: Reference point for hypervolume calculation.
                           If None, uses the minimum values for each objective.
            
        Returns:
            Hypervolume value
        """
        pareto_front = self.get_pareto_front(results)
        
        if len(pareto_front) == 0:
            return 0.0
        
        if reference_point is None:
            # Use minimum values as reference point
            reference_point = np.min(pareto_front, axis=0) - 1.0
        
        # Simple hypervolume calculation for 2D case
        if pareto_front.shape[1] == 2:
            # Sort by first objective
            sorted_indices = np.argsort(pareto_front[:, 0])
            sorted_front = pareto_front[sorted_indices]
            
            volume = 0.0
            prev_x = reference_point[0]
            
            for point in sorted_front:
                width = point[0] - prev_x
                height = point[1] - reference_point[1]
                volume += width * height
                prev_x = point[0]
            
            return volume
        
        # For higher dimensions, approximate using Monte Carlo
        return self._hypervolume_monte_carlo(pareto_front, reference_point)
    
    def _hypervolume_monte_carlo(self, pareto_front: np.ndarray, reference_point: np.ndarray, 
                                n_samples: int = 10000) -> float:
        """Approximate hypervolume using Monte Carlo sampling."""
        # Define bounding box
        max_values = np.max(pareto_front, axis=0)
        
        # Generate random points in the bounding box
        random_points = np.random.uniform(
            reference_point, max_values, (n_samples, len(reference_point))
        )
        
        # Count points dominated by the Pareto front
        dominated_count = 0
        for point in random_points:
            if self._is_dominated_by_front(point, pareto_front):
                dominated_count += 1
        
        # Calculate volume
        box_volume = np.prod(max_values - reference_point)
        return (dominated_count / n_samples) * box_volume
    
    def _is_dominated_by_front(self, point: np.ndarray, pareto_front: np.ndarray) -> bool:
        """Check if a point is dominated by any point in the Pareto front."""
        for front_point in pareto_front:
            if np.all(front_point >= point) and np.any(front_point > point):
                return True
        return False
