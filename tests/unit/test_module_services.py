import pytest
from pynomaly.domain.services.statistical_tester import StatisticalTester
from pynomaly.domain.services.model_selector import ModelSelector, ModelCandidate

# Unit tests for ModelSelector
class TestModelSelector:
    def setup_method(self):
        self.selector = ModelSelector(primary_metric="accuracy")
        self.candidates = [
            ModelCandidate("model1", "alg1", {"accuracy": 0.9}, {}, {}),
            ModelCandidate("model2", "alg2", {"accuracy": 0.8}, {}, {}),
        ]

    def test_select_best_model(self):
        best_model = self.selector.select_best_model(self.candidates)
        assert best_model["selected_model"] == "model1"

    def test_rank_models(self):
        ranked = self.selector.rank_models(self.candidates)
        assert len(ranked) == len(self.candidates)


# Unit tests for StatisticalTester
class TestStatisticalTester:
    def setup_method(self):
        self.tester = StatisticalTester(alpha=0.05)

    def test_significance_ttest(self):
        result = self.tester.test_significance({"accuracy": 0.9}, {"accuracy": 0.85})
        assert isinstance(result, dict)

    def test_comprehensive_comparison(self):
        models_metrics = {
            "model1": {"accuracy": 0.9},
            "model2": {"accuracy": 0.85}
        }
        result = self.tester.comprehensive_comparison(models_metrics)
        assert "pairwise_comparisons" in result

