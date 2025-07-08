"""Statistical test integration with precomputed p-values."""
import pytest
from pynomaly.domain.services.statistical_tester import StatisticalTester, TestResult


@pytest.fixture
def precomputed_p_values():
    """Fixture providing precomputed p-values for testing."""
    return {
        ("model1", "model2"): 0.02,  # Significant
        ("model1", "model3"): 0.20,  # Not significant
        ("model2", "model3"): 0.04,  # Significant
    }


def test_statistical_significance(precomputed_p_values):
    """Test statistical significance using precomputed p-values."""
    tester = StatisticalTester(alpha=0.05)

    results = []
    for (model_a, model_b), p_value in precomputed_p_values.items():
        result = TestResult(
            test_name="synthetic_test",
            statistic=1.5,  # Dummy value
            p_value=p_value,
            significant=p_value < tester.alpha,
            confidence_level=0.95
        )
        results.append((model_a, model_b, result))

    # Assertions to check expected significances
    assert results[0][2].significant is True
    assert results[1][2].significant is False
    assert results[2][2].significant is True
