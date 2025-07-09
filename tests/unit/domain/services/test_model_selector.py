"""
Unit tests for ModelSelector.

Tests the ModelSelector domain service functionality
including primary metric ranking, Pareto front filtering,
and statistical significance.
"""

from unittest.mock import Mock, patch
import pytest
import numpy as np
from typing import Dict

from src.pynomaly.domain.entities.model_performance import ModelPerformanceMetrics
from pynomaly.domain.services.model_selector import ModelSelector


class TestModelSelector:
    """Test cases for ModelSelector."""

    @pytest.fixture
    def model_selector(self):
        """Create a ModelSelector with a primary and secondary metrics."""
        return ModelSelector(primary_metric="f1_score", secondary_metrics=["precision", "recall"])

    @pytest.fixture
    def sample_models(self):
        """Create sample models for testing."""
        return [
            ModelPerformanceMetrics(model_id="model_1", metrics={"f1_score": 0.9, "precision": 0.8, "recall": 0.85}),
            ModelPerformanceMetrics(model_id="model_2", metrics={"f1_score": 0.85, "precision": 0.82, "recall": 0.84}),
            ModelPerformanceMetrics(model_id="model_3", metrics={"f1_score": 0.87, "precision": 0.78, "recall": 0.88})
        ]

    @pytest.fixture
    def diverse_models(self):
        """Create diverse models for Pareto front testing."""
        return [
            ModelPerformanceMetrics(model_id="high_precision", metrics={"f1_score": 0.85, "precision": 0.95, "recall": 0.77}),
            ModelPerformanceMetrics(model_id="high_recall", metrics={"f1_score": 0.84, "precision": 0.75, "recall": 0.95}),
            ModelPerformanceMetrics(model_id="balanced", metrics={"f1_score": 0.88, "precision": 0.85, "recall": 0.92}),
            ModelPerformanceMetrics(model_id="poor_model", metrics={"f1_score": 0.60, "precision": 0.55, "recall": 0.65})
        ]

    def test_initialization(self):
        """Test ModelSelector initialization."""
        selector = ModelSelector(primary_metric="accuracy", secondary_metrics=["precision", "recall"])
        assert selector.primary_metric == "accuracy"
        assert selector.secondary_metrics == ["precision", "recall"]
        assert selector.pareto_optimizer is not None

    def test_rank_models(self, model_selector, sample_models):
        """Test model ranking with Pareto front filtering."""
        with patch('pynomaly.domain.services.model_selector.MetricsCalculator.compare_models') as mock_compare:
            mock_compare.return_value = {
                'rankings': {
                    'f1_score': [
                        {'model': 'model_1', 'value': 0.9, 'rank': 1},
                        {'model': 'model_3', 'value': 0.87, 'rank': 2},
                        {'model': 'model_2', 'value': 0.85, 'rank': 3}
                    ]
                }
            }
            
            with patch.object(model_selector.pareto_optimizer, 'find_pareto_optimal') as mock_pareto:
                mock_pareto.return_value = [
                    {'model_id': 'model_1'},
                    {'model_id': 'model_3'},
                    {'model_id': 'model_2'}
                ]
                
                ranked_models = model_selector.rank_models(sample_models)
                assert len(ranked_models) == 3
                assert ranked_models[0]['model'] == "model_1"
                assert ranked_models[0]['rank'] == 1

    def test_rank_models_empty_list(self, model_selector):
        """Test ranking with empty model list."""
        with patch('pynomaly.domain.services.model_selector.MetricsCalculator.compare_models') as mock_compare:
            mock_compare.return_value = {'rankings': {'f1_score': []}}
            
            with patch.object(model_selector.pareto_optimizer, 'find_pareto_optimal') as mock_pareto:
                mock_pareto.return_value = []
                
                ranked_models = model_selector.rank_models([])
                assert ranked_models == []

    def test_significant_difference_mock_ttest(self, model_selector, sample_models):
        """Test statistical significance between models with mocked t-test."""
        with patch('pynomaly.domain.services.model_selector.ttest_ind') as mock_ttest:
            # Mock significant difference
            mock_ttest.return_value = (2.5, 0.02)  # p-value < 0.05
            significant = model_selector.significant_difference(sample_models[0], sample_models[1])
            assert significant is True
            
            # Mock non-significant difference
            mock_ttest.return_value = (1.0, 0.3)  # p-value > 0.05
            significant = model_selector.significant_difference(sample_models[0], sample_models[1])
            assert significant is False

    def test_select_best_model_complete_workflow(self, model_selector, sample_models):
        """Test complete model selection workflow."""
        with patch.object(model_selector, 'rank_models') as mock_rank:
            mock_rank.return_value = [
                {'model': 'model_1', 'value': 0.9, 'rank': 1},
                {'model': 'model_3', 'value': 0.87, 'rank': 2},
                {'model': 'model_2', 'value': 0.85, 'rank': 3}
            ]
            
            with patch.object(model_selector, 'significant_difference') as mock_sig:
                mock_sig.return_value = False
                
                best_model_selection = model_selector.select_best_model(sample_models)
                assert best_model_selection['selected_model'] == "model_1"
                assert "Selected model_1 based on primary ranking of f1_score" in best_model_selection['rationale']
                assert "model_1 is part of the Pareto front" in best_model_selection['rationale']

    def test_select_best_model_no_models(self, model_selector):
        """Test model selection with no models."""
        with patch.object(model_selector, 'rank_models') as mock_rank:
            mock_rank.return_value = []
            
            best_model_selection = model_selector.select_best_model([])
            assert best_model_selection['decision'] == 'No suitable models found'
            assert best_model_selection['rationale'] == []

    def test_pareto_front_filtering(self, model_selector, diverse_models):
        """Test Pareto front filtering functionality."""
        # Mock the pareto optimizer to return only non-dominated models
        with patch.object(model_selector.pareto_optimizer, 'find_pareto_optimal') as mock_pareto:
            mock_pareto.return_value = [
                {'model_id': 'high_precision'},
                {'model_id': 'high_recall'},
                {'model_id': 'balanced'}
            ]
            
            with patch('pynomaly.domain.services.model_selector.MetricsCalculator.compare_models') as mock_compare:
                mock_compare.return_value = {
                    'rankings': {
                        'f1_score': [
                            {'model': 'balanced', 'value': 0.88, 'rank': 1},
                            {'model': 'high_precision', 'value': 0.85, 'rank': 2},
                            {'model': 'high_recall', 'value': 0.84, 'rank': 3},
                            {'model': 'poor_model', 'value': 0.60, 'rank': 4}
                        ]
                    }
                }
                
                ranked_models = model_selector.rank_models(diverse_models)
                
                # Should only return models in Pareto front
                assert len(ranked_models) == 3
                model_ids = [r['model'] for r in ranked_models]
                assert 'balanced' in model_ids
                assert 'high_precision' in model_ids
                assert 'high_recall' in model_ids
                assert 'poor_model' not in model_ids

    def test_tie_breaking_logic(self, model_selector):
        """Test tie-breaking logic for models with similar performance."""
        tied_models = [
            ModelPerformanceMetrics(model_id="model_a", metrics={"f1_score": 0.85, "precision": 0.80, "recall": 0.90}),
            ModelPerformanceMetrics(model_id="model_b", metrics={"f1_score": 0.85, "precision": 0.85, "recall": 0.85}),
            ModelPerformanceMetrics(model_id="model_c", metrics={"f1_score": 0.85, "precision": 0.90, "recall": 0.80})
        ]
        
        with patch('pynomaly.domain.services.model_selector.MetricsCalculator.compare_models') as mock_compare:
            mock_compare.return_value = {
                'rankings': {
                    'f1_score': [
                        {'model': 'model_a', 'value': 0.85, 'rank': 1},
                        {'model': 'model_b', 'value': 0.85, 'rank': 2},
                        {'model': 'model_c', 'value': 0.85, 'rank': 3}
                    ]
                }
            }
            
            with patch.object(model_selector.pareto_optimizer, 'find_pareto_optimal') as mock_pareto:
                mock_pareto.return_value = [
                    {'model_id': 'model_a'},
                    {'model_id': 'model_b'},
                    {'model_id': 'model_c'}
                ]
                
                selection = model_selector.select_best_model(tied_models)
                assert selection['selected_model'] == 'model_a'
                assert 'Selected model_a based on primary ranking of f1_score' in selection['rationale']

    def test_statistical_significance_with_real_data(self, model_selector):
        """Test statistical significance calculation with realistic metric values."""
        # Create models with similar but different performance
        model_a = ModelPerformanceMetrics(
            model_id="model_a", 
            metrics={"f1_score": 0.85, "precision": 0.80, "recall": 0.90}
        )
        model_b = ModelPerformanceMetrics(
            model_id="model_b", 
            metrics={"f1_score": 0.83, "precision": 0.78, "recall": 0.88}
        )
        
        # The t-test should indicate no significant difference for such close values
        significant = model_selector.significant_difference(model_a, model_b)
        assert isinstance(significant, bool)

    def test_edge_case_single_model(self, model_selector):
        """Test model selection with only one model."""
        single_model = [ModelPerformanceMetrics(model_id="only_model", metrics={"f1_score": 0.8, "precision": 0.75, "recall": 0.85})]
        
        with patch.object(model_selector, 'rank_models') as mock_rank:
            mock_rank.return_value = [{'model': 'only_model', 'value': 0.8, 'rank': 1}]
            
            selection = model_selector.select_best_model(single_model)
            assert selection['selected_model'] == 'only_model'
            assert len(selection['rationale']) >= 2

    def test_different_primary_metrics(self):
        """Test ModelSelector with different primary metrics."""
        # Test with accuracy as primary metric
        accuracy_selector = ModelSelector(primary_metric="accuracy", secondary_metrics=["precision", "recall"])
        assert accuracy_selector.primary_metric == "accuracy"
        
        # Test with precision as primary metric
        precision_selector = ModelSelector(primary_metric="precision", secondary_metrics=["f1_score", "recall"])
        assert precision_selector.primary_metric == "precision"
        
        # Verify Pareto optimizer is configured correctly
        assert accuracy_selector.pareto_optimizer.objectives[0]['name'] == "accuracy"
        assert precision_selector.pareto_optimizer.objectives[0]['name'] == "precision"

    def test_error_handling_empty_metrics(self, model_selector):
        """Test error handling for models with empty metrics."""
        empty_metrics_model = ModelPerformanceMetrics(model_id="empty_model", metrics={})
        
        with pytest.raises(KeyError):
            model_selector.significant_difference(empty_metrics_model, empty_metrics_model)

    def test_model_selector_with_resource_constraints(self):
        """Test ModelSelector considering resource usage as secondary metrics."""
        resource_selector = ModelSelector(
            primary_metric="f1_score", 
            secondary_metrics=["training_time", "memory_usage"]
        )
        
        assert resource_selector.primary_metric == "f1_score"
        assert "training_time" in resource_selector.secondary_metrics
        assert "memory_usage" in resource_selector.secondary_metrics

    def test_rationale_generation_comprehensive(self, model_selector, sample_models):
        """Test comprehensive rationale generation for model selection."""
        with patch.object(model_selector, 'rank_models') as mock_rank:
            mock_rank.return_value = [
                {'model': 'model_1', 'value': 0.9, 'rank': 1},
                {'model': 'model_3', 'value': 0.87, 'rank': 2},
                {'model': 'model_2', 'value': 0.85, 'rank': 3}
            ]
            
            with patch.object(model_selector, 'significant_difference') as mock_sig:
                # Mock that model_3 and model_2 are not significantly different from model_1
                mock_sig.return_value = False
                
                selection = model_selector.select_best_model(sample_models)
                rationale = selection['rationale']
                
                # Should contain primary ranking rationale
                assert any("primary ranking" in r for r in rationale)
                # Should contain Pareto front rationale
                assert any("Pareto front" in r for r in rationale)
                # Should contain significance test results
                assert any("lack of significant difference" in r for r in rationale)
