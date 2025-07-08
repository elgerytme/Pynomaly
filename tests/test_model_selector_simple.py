#!/usr/bin/env python3
"""Simple standalone test for ModelSelector functionality."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from datetime import datetime
from uuid import uuid4
from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np
from scipy.stats import ttest_ind

# Mock dependencies
@dataclass
class ModelPerformanceMetrics:
    """Mock ModelPerformanceMetrics for testing."""
    model_id: str
    metrics: Dict[str, float]
    timestamp: datetime = None
    id: str = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.id is None:
            self.id = str(uuid4())
        if self.metadata is None:
            self.metadata = {}

class MockParetoOptimizer:
    """Mock ParetoOptimizer for testing."""
    def __init__(self, objectives):
        self.objectives = objectives
    
    def find_pareto_optimal(self, results):
        # Simple mock: return all results as Pareto optimal
        return results

class MockMetricsCalculator:
    """Mock MetricsCalculator for testing."""
    @staticmethod
    def compare_models(model_results, primary_metric):
        # Sort by primary metric
        sorted_models = sorted(
            model_results.items(),
            key=lambda x: x[1].get(primary_metric, 0),
            reverse=True
        )
        
        rankings = []
        for i, (model_id, metrics) in enumerate(sorted_models):
            rankings.append({
                'model': model_id,
                'value': metrics.get(primary_metric, 0),
                'rank': i + 1
            })
        
        return {'rankings': {primary_metric: rankings}}

# ModelSelector implementation
class ModelSelector:
    """Comprehensive model ranking and selection engine."""

    def __init__(self, primary_metric: str, secondary_metrics: List[str]):
        self.primary_metric = primary_metric
        self.secondary_metrics = secondary_metrics
        self.pareto_optimizer = MockParetoOptimizer(
            objectives=[{'name': primary_metric, 'direction': 'max'}] +
                      [{'name': metric, 'direction': 'max'} for metric in secondary_metrics]
        )

    def rank_models(self, models: List[ModelPerformanceMetrics]) -> List[Dict[str, Any]]:
        """Rank models based on primary metric and apply Pareto front filtering."""
        if not models:
            return []
            
        model_dicts = [{
            'model_id': model.model_id,
            'metrics': model.metrics
        } for model in models]

        # Primary Metric Ranking
        ranked_results = MockMetricsCalculator.compare_models(
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
        """Test statistical significance between two models."""
        a_metrics = np.array(list(model_a.metrics.values()))
        b_metrics = np.array(list(model_b.metrics.values()))

        stat, p_value = ttest_ind(a_metrics, b_metrics, equal_var=False)
        return p_value < 0.05

    def select_best_model(self, models: List[ModelPerformanceMetrics]) -> Dict[str, Any]:
        """Select the best model based on comprehensive criteria."""
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

        return {
            'selected_model': best_model,
            'rationale': rationale,
            'ranking_details': ranked_models
        }

def test_model_selector():
    """Test the ModelSelector functionality."""
    print("Testing ModelSelector...")
    
    # Create test models
    models = [
        ModelPerformanceMetrics(
            model_id="model_1", 
            metrics={"f1_score": 0.9, "precision": 0.8, "recall": 0.85}
        ),
        ModelPerformanceMetrics(
            model_id="model_2", 
            metrics={"f1_score": 0.85, "precision": 0.82, "recall": 0.84}
        ),
        ModelPerformanceMetrics(
            model_id="model_3", 
            metrics={"f1_score": 0.87, "precision": 0.78, "recall": 0.88}
        )
    ]
    
    # Test initialization
    selector = ModelSelector(
        primary_metric="f1_score", 
        secondary_metrics=["precision", "recall"]
    )
    assert selector.primary_metric == "f1_score"
    assert selector.secondary_metrics == ["precision", "recall"]
    print("✓ Initialization test passed")
    
    # Test ranking
    ranked = selector.rank_models(models)
    assert len(ranked) == 3
    assert ranked[0]['model'] == "model_1"  # Highest f1_score
    assert ranked[0]['value'] == 0.9
    print("✓ Ranking test passed")
    
    # Test empty models
    empty_ranked = selector.rank_models([])
    assert empty_ranked == []
    print("✓ Empty models test passed")
    
    # Test significance testing
    sig_diff = selector.significant_difference(models[0], models[1])
    assert isinstance(sig_diff, bool)
    print("✓ Significance test passed")
    
    # Test best model selection
    result = selector.select_best_model(models)
    assert result['selected_model'] == "model_1"
    assert len(result['rationale']) >= 2
    assert "Selected model_1 based on primary ranking of f1_score" in result['rationale']
    assert "model_1 is part of the Pareto front" in result['rationale']
    print("✓ Best model selection test passed")
    
    # Test with no models
    no_models_result = selector.select_best_model([])
    assert no_models_result['decision'] == 'No suitable models found'
    assert no_models_result['rationale'] == []
    print("✓ No models test passed")
    
    print("\nAll tests passed! ✓")
    print(f"Best model selected: {result['selected_model']}")
    print("Rationale:")
    for reason in result['rationale']:
        print(f"  - {reason}")

if __name__ == "__main__":
    test_model_selector()
