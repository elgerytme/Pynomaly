"""
Unit tests for ModelSelector.

Tests the ModelSelector domain service functionality
including primary metric ranking, Pareto front filtering,
and statistical significance.
"""

from unittest.mock import Mock
import pytest
from typing import Dict

from src.pynomaly.domain.entities.model_performance import ModelPerformanceMetrics
from src.pynomaly.domain.services.model_selector import ModelSelector


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

    def test_rank_models(self, model_selector, sample_models):
        """Test model ranking with Pareto front filtering."""
        ranked_models = model_selector.rank_models(sample_models)
        assert len(ranked_models) == 3
        assert ranked_models[0]['model'] == "model_1"

    def test_significant_difference(self, model_selector, sample_models):
        """Test statistical significance between models."""
        significant = model_selector.significant_difference(sample_models[0], sample_models[1])
        assert significant is False  # Assuming mock t-test returns no significance

    def test_select_best_model(self, model_selector, sample_models):
        """Test selection of the best model with rationale."""
        best_model_selection = model_selector.select_best_model(sample_models)
        assert best_model_selection['selected_model'] == "model_1"
        assert "Selected model_1 based on primary ranking of f1_score" in best_model_selection['rationale']

